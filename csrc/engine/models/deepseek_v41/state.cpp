#include "engine/models/deepseek_v41/state.h"
#include "engine/models/deepseek_v41/ops.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/deepseek_v41/snapshot.h"
#include "engine/models/common/check.h"
#include "engine/models/session/state.h"

#include <algorithm>
#include <stdexcept>

namespace tyloi::cuda::deepseek_v41_runtime {
SharedState::SharedState(const CommonConfig& config, const ModelTables& tables)
    : config(config), tables(tables), layers(static_cast<std::size_t>(config.n_layers)),
      engram_hash(std::make_unique<EngramHashState>(tables.engram_parameters)),
      engram_states(static_cast<std::size_t>(config.n_layers)) {
    for (const auto* memory : tables.engram_prefetch)
        engram_states.at(static_cast<std::size_t>(memory->layer_id())) = memory->make_state();
}
} // namespace tyloi::cuda::deepseek_v41_runtime

namespace tyloi::cuda::deepseek_v41 {

using deepseek_v41_runtime::Block;

namespace {
bool session_tensor(const ::tyloi::Tensor& value,int64_t tokens,int64_t width,
                    ::tyloi::ScalarType dtype) {
    return value && value.defined() && value.is_cuda() && value.dim()==3 && value.size(0)==1 &&
        value.size(1)==tokens && value.size(2)==width && value.scalar_type()==dtype &&
        value.representation<DenseTensor>();
}

void validate_layer(const Block& block,const tyloi::models::deepseek_v41::BlockSnapshot& saved,
                    int64_t position) {
    const auto& config=block.config;
    const bool local=session_tensor(saved.local_kv,config.sliding_window,config.head_dim,
                                     ::tyloi::kFloat16);
    bool global=saved.compressed_length==0 && saved.partial_length==0 &&
        !saved.compressed_kv && !saved.index_k && !saved.partial_kv && !saved.partial_score;
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
        else global=global && !saved.partial_kv && !saved.partial_score;
    }
    if(!local || !global)throw CudaSessionStateError("DeepSeek-V4.1 session layer geometry is incompatible");
}
}

tyloi::models::SnapshotSize snapshot_size(const State& state,const tyloi::models::Mtp* mtp) {
    std::size_t incoming=256; // Rolling Engram history and scalar metadata.
    const auto prefix_bytes=[](const auto& tensor,int64_t rows) -> std::size_t {
        return tensor.defined()?std::size_t(tensor.size(0))*rows*tensor.size(2)*tensor.element_size():0;
    };
    for(const auto& item:state.model_.blocks) {
        TYLOINFER_RUNTIME_CHECK(state.shared && item->layer >= 0 &&
                                  item->layer < std::int64_t(state.shared->layers.size()),
                              "DeepSeek-V4.1 layer state is not allocated");
        const auto& layer=state.shared->layers[static_cast<std::size_t>(item->layer)];
        incoming+=session_tensor_bytes(layer.local_kv)+session_tensor_bytes(layer.partial_kv)+
            session_tensor_bytes(layer.partial_score)+prefix_bytes(layer.compressed_kv,layer.compressed_length)+
            prefix_bytes(layer.index_k,layer.compressed_length);
    }
    if(mtp) {
        const auto& c=state.shared->config;
        incoming+=std::size_t(c.sliding_window)*sizeof(mfq_half)*
            (c.n_mtp_layers*c.head_dim+c.hidden*c.dspark_target_layer_ids.size());
    }
    return {incoming,incoming};
}

State::State(Model& model)
    : shared(std::make_unique<deepseek_v41_runtime::SharedState>(model.config, model.tables)), model_(model) {}

State::~State() = default;

void State::set_image_mask(::tyloi::Tensor mask) {
    auto& state = *shared;
    if (!mask) {
        state.image_mask = {}; state.text_rows = {}; state.routing_mask = {}; state.text_row_ids.clear();
        return;
    }
    const auto& dense = mask.dense();
    TYLOINFER_RUNTIME_CHECK(dense.is_cpu() && dense.dim()==2 && dense.is_contiguous() &&
                            dense.scalar_type()==::tyloi::kUInt8,
                            "DeepSeek-V4.1 image mask must be contiguous CPU uint8 [B,T]");
    std::vector<int64_t> images;
    state.text_row_ids.clear();
    const auto* flags=dense.data_ptr<uint8_t>();
    for(int64_t i=0;i<dense.numel();++i)(flags[i]?images:state.text_row_ids).push_back(i);
    if(images.empty()) { set_image_mask({}); return; }
    auto options=::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kInt64);
    auto text=::tyloi::cuda::tensor(state.text_row_ids,options);
    auto routing=dense.reshape({-1,1}).to(::tyloi::kCUDA).to(::tyloi::kBool);
    state.image_mask=dense; state.text_rows=std::move(text); state.routing_mask=std::move(routing);
}

void State::begin_forward(bool capture_raw_hidden) {
    shared->capture_dspark_targets = capture_raw_hidden;
    shared->dspark_target_hiddens.clear();
    if (capture_raw_hidden)
        shared->dspark_target_hiddens.resize(model_.config.dspark_target_layer_ids.size());
}

void State::reset(std::int64_t next_batch) {
    cache_pos = decode_position_delta = 0;
    speculative_start = -1;
    speculative_confirmed = 0;
    speculative_suffix_forward = false;
    batch = next_batch;
    for (auto& block : model_.blocks)
        block->reset(*shared, next_batch);
}

std::vector<::tyloi::Tensor> State::storage() const {
    std::vector<::tyloi::Tensor> result;
    for (const auto& block : model_.blocks) {
        const auto& state = shared->layers.at(static_cast<std::size_t>(block->layer));
        if (state.local_kv.defined()) result.push_back(std::make_shared<tyloi::DenseTensor>(state.local_kv));
        if (state.compressed_kv.defined()) result.push_back(std::make_shared<tyloi::DenseTensor>(state.compressed_kv));
        if (state.index_k.defined()) result.push_back(std::make_shared<tyloi::DenseTensor>(state.index_k));
        if (state.partial_kv.defined()) result.push_back(std::make_shared<tyloi::DenseTensor>(state.partial_kv));
        if (state.partial_score.defined()) result.push_back(state.partial_score);
        if (state.speculative_local_slots.defined()) result.push_back(state.speculative_local_slots);
        if (state.speculative_slot_ids.defined()) result.push_back(state.speculative_slot_ids);
        if (state.speculative_partial_kv.defined()) result.push_back(state.speculative_partial_kv);
        if (state.speculative_partial_score.defined()) result.push_back(state.speculative_partial_score);
        if (state.speculative_projected_kv.defined()) result.push_back(state.speculative_projected_kv);
        if (state.speculative_projected_score.defined()) result.push_back(state.speculative_projected_score);
        if (state.current_ids.defined()) result.push_back(state.current_ids);
    }
    if (shared->compressed_kv.defined()) result.push_back(shared->compressed_kv);
    if (shared->index_k.defined()) result.push_back(shared->index_k);
    if (shared->topk.defined()) result.push_back(shared->topk);
    if (shared->candidate_blocks.defined()) result.push_back(shared->candidate_blocks);
    if (shared->previous_pre.defined()) result.push_back(shared->previous_pre);
#ifndef TYLOINFER_NATIVE_CUDA
    if (shared->attention_meta.defined()) result.push_back(shared->attention_meta);
#endif
    if (shared->image_mask.defined()) result.push_back(shared->image_mask);
    if (shared->text_rows.defined()) result.push_back(shared->text_rows);
    if (shared->routing_mask.defined()) result.push_back(shared->routing_mask);
    for (const auto& hidden : shared->dspark_target_hiddens)
        if (hidden.defined()) result.push_back(hidden);
    for (const auto& engram : shared->engram_states) {
        if (!engram) continue;
        auto buffers = engram->storage();
        result.insert(result.end(), buffers.begin(), buffers.end());
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
    TYLOINFER_RUNTIME_CHECK(shared->engram_hash, "DeepSeek-V4.1 Engram state is unavailable");
    shared->engram_hash->begin_speculative();
    std::size_t begun = 0;
    try {
        for (auto& block : model_.blocks) {
            MfqCudaGuard guard(block->cuda_device);
            block->begin_speculative(*shared, draft_tokens);
            ++begun;
        }
    } catch (...) {
        for (std::size_t i = 0; i < begun; ++i) {
            try {
                MfqCudaGuard guard(model_.blocks[i]->cuda_device);
                model_.blocks[i]->commit_speculative(*shared);
            } catch (...) {
            }
        }
        try {
            shared->engram_hash->rollback_speculative(cache_pos);
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
        block->commit_speculative(*shared);
    }
    if (shared->engram_hash) shared->engram_hash->commit_speculative();
    speculative_start = -1;
    speculative_confirmed = 0;
}

void State::rollback_speculative(std::int64_t accepted) {
    using tyloi::models::require_model;
    require_model(speculative_start >= 0 && accepted >= 0, "no speculative transaction");
    const auto keep = speculative_start + speculative_confirmed + accepted;
    for (auto& block : model_.blocks)
        block->rollback_speculative(*shared, keep);
    TYLOINFER_RUNTIME_CHECK(shared->engram_hash, "DeepSeek-V4.1 Engram state is unavailable");
    shared->engram_hash->rollback_speculative(keep);
    cache_pos = keep;
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
        const auto& source=shared->layers.at(static_cast<std::size_t>(block.layer));
        if(source.speculative_pending || source.position!=cache_pos)
            throw std::runtime_error("DeepSeek-V4.1 session attention state is not committed");
        tyloi::models::deepseek_v41::BlockSnapshot saved;
        const auto copy=[&](const DenseTensor& value) -> ::tyloi::Tensor {
            if(!value.defined())return {};
            auto result=std::make_shared<DenseTensor>(value.clone());
            state.bytes+=session_tensor_bytes(*result);return result;
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
        const auto& saved=snapshot->blocks[i];auto& target=shared->layers.at(i);
#ifdef TYLOINFER_NATIVE_CUDA
        auto local_kv=restore_session_tensor(target.local_kv,saved.local_kv);
        target.local_kv=local_kv.defined()?local_kv.dense():DenseTensor{};
#else
        restore_session_tensor(target.local_kv,saved.local_kv);
#endif
        if(block.kv_source()) {
#ifdef TYLOINFER_NATIVE_CUDA
            target.compressed_kv=restore_session_prefix_tensor(target.compressed_kv,saved.compressed_kv,1,
                                          std::max<int64_t>(1,block.max_context/block.ratio)).dense();
            target.index_k=restore_session_prefix_tensor(target.index_k,saved.index_k,1,
                                          std::max<int64_t>(1,block.max_context/block.ratio)).dense();
#else
            restore_session_prefix_tensor(target.compressed_kv,saved.compressed_kv,1,
                                          std::max<int64_t>(1,block.max_context/block.ratio));
            restore_session_prefix_tensor(target.index_k,saved.index_k,1,
                                          std::max<int64_t>(1,block.max_context/block.ratio));
#endif
            if(block.ratio>1) {
#ifdef TYLOINFER_NATIVE_CUDA
                auto partial_kv=restore_session_tensor(target.partial_kv,saved.partial_kv);
                target.partial_kv=partial_kv.defined()?partial_kv.dense():DenseTensor{};
                auto partial_score=restore_session_tensor(target.partial_score,saved.partial_score);
                target.partial_score=partial_score.defined()?partial_score.dense():DenseTensor{};
#else
                restore_session_tensor(target.partial_kv,saved.partial_kv);
                restore_session_tensor(target.partial_score,saved.partial_score);
#endif
            }
        }
        target.position=state.cache_pos;target.compressed_length=saved.compressed_length;
        target.partial_length=saved.partial_length;
    }
    *shared->engram_hash=std::move(hash);
    set_image_mask({});
    cache_pos=state.cache_pos;
}

} // namespace tyloi::cuda::deepseek_v41
