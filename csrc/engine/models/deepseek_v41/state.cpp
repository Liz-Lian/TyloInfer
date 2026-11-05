#include "engine/models/deepseek_v41/state.h"
#include "engine/models/deepseek_v41/ops.h"
#include "engine/models/session.h"
#include "engine/models/deepseek_v41/snapshot.h"
#include "engine/models/common/check.h"

#include <algorithm>
#include <stdexcept>

namespace tyloi::cuda::deepseek_v41 {

using deepseek_v41_runtime::Block;

namespace {
bool session_tensor(const DenseTensor& value,int64_t tokens,int64_t width,
                    ::tyloi::ScalarType dtype) {
    return value.defined() && value.is_cuda() && value.dim()==3 && value.size(0)==1 &&
        value.size(1)==tokens && value.size(2)==width && value.scalar_type()==dtype;
}

void validate_layer(const Block& block,const tyloi::models::deepseek_v41::BlockSnapshot& saved,
                    int64_t position) {
    const auto& config=block.config;
    const bool local=session_tensor(saved.local_kv,config.sliding_window,config.head_dim,
                                     ::tyloi::kFloat16);
    bool global=saved.compressed_length==0 && saved.partial_length==0 &&
        !saved.compressed_kv.defined() && !saved.index_k.defined() &&
        !saved.partial_kv.defined() && !saved.partial_score.defined();
    if(block.kv_source()) {
        global=block.ratio>0 && saved.compressed_length==position/block.ratio &&
            saved.partial_length==(block.ratio>1?position%block.ratio:0) &&
            session_tensor(saved.compressed_kv,saved.compressed_length,config.head_dim,
                           ::tyloi::kFloat16) &&
            session_tensor(saved.index_k,saved.compressed_length,config.index_head_dim,
                           ::tyloi::kFloat16);
        if(block.ratio>1)
            global=global && session_tensor(saved.partial_kv,block.ratio,config.head_dim,
                                            ::tyloi::kFloat32) &&
                session_tensor(saved.partial_score,block.ratio,config.head_dim,
                               ::tyloi::kFloat32);
        else global=global && !saved.partial_kv.defined() && !saved.partial_score.defined();
    }
    if(!local || !global)throw CudaSessionStateError("DeepSeek-V4.1 session layer geometry is incompatible");
}
}

State::State(Model& model) : shared(model.shared), model_(model) {
    model_.active = this;
    TYLOINFER_RUNTIME_CHECK(shared, "DeepSeek-V4.1 model has no shared state");
}

State::~State() {
    if (model_.active == this)
        model_.active = nullptr;
}

void State::reset(std::int64_t next_batch) {
    cache_pos = decode_position_delta = 0;
    speculative_start = -1;
    speculative_confirmed = 0;
    speculative_suffix_forward = false;
    batch = next_batch;
    for (auto& block : model_.blocks)
        block->reset(next_batch);
}

std::vector<tyloi::DenseTensor> State::storage() const {
    std::vector<tyloi::DenseTensor> result;
    for (const auto& block : model_.blocks) {
        const auto& state = block->L();
        if (state.local_kv.defined()) result.push_back(state.local_kv);
        if (state.compressed_kv.defined()) result.push_back(state.compressed_kv);
        if (state.index_k.defined()) result.push_back(state.index_k);
        if (state.partial_kv.defined()) result.push_back(state.partial_kv);
    }
    return result;
}

void State::begin_speculative_suffix(std::int64_t draft_tokens) {
    using tyloi::models::require_model;
    require_model(model_.supports_suffix_speculation() && draft_tokens > 0 &&
                      cache_pos + draft_tokens <= model_.max_position_embeddings(),
                  "invalid speculative suffix");
    speculative_start = cache_pos;
    speculative_confirmed = 0;
    model_.adapter_begin_speculative();
    std::size_t begun = 0;
    try {
        for (auto& block : model_.blocks) {
            MfqCudaGuard guard(block->cuda_device);
            block->begin_speculative(draft_tokens);
            ++begun;
        }
    } catch (...) {
        for (std::size_t i = 0; i < begun; ++i) {
            try {
                MfqCudaGuard guard(model_.blocks[i]->cuda_device);
                model_.blocks[i]->commit_speculative();
            } catch (...) {
            }
        }
        try {
            model_.adapter_rollback_speculative(cache_pos);
        } catch (...) {
        }
        speculative_start = -1;
        throw;
    }
}

void State::commit_speculative() {
    using tyloi::models::require_model;
    require_model(speculative_start >= 0, "no speculative transaction");
    for (auto& block : model_.blocks) {
        MfqCudaGuard guard(block->cuda_device);
        block->commit_speculative();
    }
    model_.adapter_commit_speculative();
    speculative_start = -1;
    speculative_confirmed = 0;
}

bool State::supports_text_session_state() const {
    return text_session_state_kind() != TextSessionStateKind::Unsupported;
}

bool State::supports_paged_text_session_state() const {
    return text_session_state_kind() == TextSessionStateKind::DeepseekV41;
}

TextSessionStateKind State::text_session_state_kind() const {
    return shared && shared->engram_hash && !model_.blocks.empty()
        ? TextSessionStateKind::DeepseekV41
        : TextSessionStateKind::Unsupported;
}

TextSessionState State::capture_text_session_state(const std::vector<std::int64_t>& tokens) const {
    if(text_session_state_kind()!=TextSessionStateKind::DeepseekV41 || cache_pos<=0 ||
            std::size_t(cache_pos)!=tokens.size())
        throw std::runtime_error("DeepSeek-V4.1 text session token count disagrees with cache");
    auto hash=shared->engram_hash->snapshot();
    if(hash.batch!=1 || hash.position!=cache_pos)
        throw std::runtime_error("DeepSeek-V4.1 session Engram position disagrees with cache");
    TextSessionState state;
    state.tokens=tokens;state.cache_pos=cache_pos;
    auto& snapshot=state.payload.emplace<tyloi::models::deepseek_v41::Snapshot>();
    snapshot.engram=std::make_shared<const tyloi::models::deepseek_v41::EngramHashSnapshot>(std::move(hash));
    state.bytes=snapshot.engram->context.size()*sizeof(int64_t);
    auto& layers=snapshot.blocks;
    layers.reserve(model_.blocks.size());
    for(const auto& item:model_.blocks) {
        const Block& block=*item;
        MfqCudaGuard guard(block.cuda_device);
        const auto& source=block.L();
        if(block.L().speculative_pending || source.position!=cache_pos)
            throw std::runtime_error("DeepSeek-V4.1 session attention state is not committed");
        tyloi::models::deepseek_v41::BlockSnapshot saved;
        const auto copy=[&](const DenseTensor& value) {
            if(!value.defined())return DenseTensor{};
            auto result=value.clone();state.bytes+=session_tensor_bytes(result);return result;
        };
        saved.local_kv=copy(source.local_kv);
        saved.compressed_length=source.compressed_length;saved.partial_length=source.partial_length;
        if(source.compressed_kv.defined())
            saved.compressed_kv=copy(source.compressed_kv.narrow(1,0,source.compressed_length));
        if(source.index_k.defined())saved.index_k=copy(source.index_k.narrow(1,0,source.compressed_length));
        saved.partial_kv=copy(source.partial_kv);saved.partial_score=copy(source.partial_score);
        validate_layer(block,saved,state.cache_pos);
        layers.push_back(std::move(saved));
    }
    return state;
}

void State::restore_text_session_state(const TextSessionState& state) {
    const auto* snapshot=std::get_if<tyloi::models::deepseek_v41::Snapshot>(&state.payload);
    if(text_session_state_kind()!=TextSessionStateKind::DeepseekV41 || !snapshot ||
            snapshot->blocks.size()!=model_.blocks.size() ||
            state.cache_pos<=0 || state.cache_pos>model_.max_position_embeddings() ||
            std::size_t(state.cache_pos)!=state.tokens.size() || state.decode_position_delta!=0 ||
            !snapshot->engram || snapshot->engram->batch!=1 ||
            snapshot->engram->position!=state.cache_pos)
        throw CudaSessionStateError("DeepSeek-V4.1 text session state is incompatible");
    // Validate every layer and rolling n-gram before changing the active cache.
    auto hash=*shared->engram_hash;
    hash.restore(*snapshot->engram);
    for(std::size_t i=0;i<snapshot->blocks.size();++i)
        validate_layer(*model_.blocks[i],snapshot->blocks[i],state.cache_pos);
    reset(1);
    for(std::size_t i=0;i<snapshot->blocks.size();++i) {
        Block& block=*model_.blocks[i];
        MfqCudaGuard guard(block.cuda_device);
        const auto& saved=snapshot->blocks[i];auto& target=block.L();
        restore_session_tensor(target.local_kv,saved.local_kv);
        if(block.kv_source()) {
            restore_session_prefix_tensor(target.compressed_kv,saved.compressed_kv,1,
                                          std::max<int64_t>(1,block.max_context/block.ratio));
            restore_session_prefix_tensor(target.index_k,saved.index_k,1,
                                          std::max<int64_t>(1,block.max_context/block.ratio));
            if(block.ratio>1) {
                restore_session_tensor(target.partial_kv,saved.partial_kv);
                restore_session_tensor(target.partial_score,saved.partial_score);
            }
        }
        target.position=state.cache_pos;target.compressed_length=saved.compressed_length;
        target.partial_length=saved.partial_length;
    }
    *shared->engram_hash=std::move(hash);
    shared->set_image_mask({});
    cache_pos=state.cache_pos;
}

} // namespace tyloi::cuda::deepseek_v41
