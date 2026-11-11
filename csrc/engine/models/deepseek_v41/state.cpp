#include "engine/models/deepseek_v41/state.h"
#include "engine/models/deepseek_v41/ops.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/deepseek_v41/snapshot.h"
#include "engine/models/common/check.h"
#include "engine/models/session/state.h"
#include "engine/ops.h"

#include <algorithm>
#include <limits>
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

namespace {
#ifdef TYLOINFER_NATIVE_CUDA
using Growth = std::vector<std::pair<tyloi::Device, std::size_t>>;
std::size_t mul(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("DeepSeek-V4.1 allocation planning overflow");
    return a * b;
}
void add(Growth& result, tyloi::Device device, std::size_t bytes) {
    if (!bytes) return;
    auto it=std::find_if(result.begin(),result.end(),[device](const auto& item){return item.first==device;});
    if(it==result.end()) result.emplace_back(device,bytes);
    else {
        if(it->second>std::numeric_limits<std::size_t>::max()-bytes)
            throw std::overflow_error("DeepSeek-V4.1 allocation planning overflow");
        it->second+=bytes;
    }
}
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA
std::vector<std::pair<tyloi::Device, std::size_t>> State::reset_growth(std::int64_t next_batch) const {
    if(next_batch<=0) throw std::invalid_argument("DeepSeek-V4.1 reset batch must be positive");
    Growth result;
    for(const auto& item:model_.blocks) {
        const auto& b=*item;
        const auto& layer=shared->layers.at(static_cast<std::size_t>(b.layer));
        const tyloi::Device device(tyloi::DeviceType::cuda,b.cuda_device);
        const auto plan=[&](const tyloi::Tensor& target,std::int64_t length,std::int64_t width,tyloi::ScalarType type,std::size_t element) {
            const bool same=target.defined() && target.representation<DenseTensor>() && target.dense().is_contiguous() && target.device()==device &&
                target.scalar_type()==type && target.dim()==3 && target.size(0)==next_batch &&
                target.size(1)==length && target.size(2)==width;
            if(!same) add(result,device,mul(mul(mul(std::size_t(next_batch),std::size_t(length)),std::size_t(width)),element));
        };
        plan(layer.local_kv,b.config.sliding_window,b.config.head_dim,tyloi::kFloat16,2);
        if(b.kv_source()) {
            const auto capacity=std::max<std::int64_t>(1,b.max_context/b.ratio);
            plan(layer.compressed_kv,capacity,b.config.head_dim,tyloi::kFloat16,2);
            plan(layer.index_k,capacity,b.config.index_head_dim,tyloi::kFloat16,2);
            if(b.ratio>1) {
                plan(layer.partial_kv,b.ratio,b.config.head_dim,tyloi::kFloat32,4);
                plan(layer.partial_score,b.ratio,b.config.head_dim,tyloi::kFloat32,4);
            }
        }
    }
    return result;
}
std::vector<std::pair<tyloi::Device, std::size_t>> State::suffix_growth(std::int64_t tokens) const {
    if(batch!=1 || tokens<=0 || speculative_start>=0)
        throw std::invalid_argument("DeepSeek-V4.1 invalid speculative allocation window");
    Growth result;
    for(const auto& item:model_.blocks) {
        const auto& b=*item;
        const auto& layer=shared->layers.at(static_cast<std::size_t>(b.layer));
        if(tokens>b.config.sliding_window || layer.speculative_pending || !layer.local_kv.defined())
            throw std::invalid_argument("DeepSeek-V4.1 invalid speculative checkpoint");
        const auto device=tyloi::Device(tyloi::DeviceType::cuda,b.cuda_device);
        const auto rows=mul(std::size_t(batch),std::size_t(tokens));
        add(result,device,mul(mul(std::size_t(tokens),sizeof(std::int64_t)),3));
        add(result,device,mul(mul(rows,std::size_t(b.config.head_dim)),4));
        if(layer.partial_kv.defined())
            add(result,device,mul(mul(mul(std::size_t(batch),std::size_t(b.ratio)),std::size_t(b.config.head_dim)),8));
    }
    return result;
}

std::vector<std::pair<tyloi::Device, std::size_t>> State::growth(
    std::int64_t tokens,const tyloi::models::ForwardOptions&,std::int64_t confirmed) const {
    if(batch<=0 || tokens<=0 || confirmed<0 || confirmed>=tokens)
        throw std::invalid_argument("DeepSeek-V4.1 invalid forward allocation window");
    Growth result=confirmed>0 ? suffix_growth(tokens) : Growth{};
    if(confirmed>0 || speculative_start>=0) {
        for(const auto& item:model_.blocks) {
            const auto& b=*item;
            const auto device=tyloi::Device(tyloi::DeviceType::cuda,b.cuda_device);
            const auto rows=mul(std::size_t(batch),std::size_t(tokens));
            if(b.kv_source() && b.ratio>1) {
                add(result,device,mul(mul(rows,std::size_t(b.config.head_dim)),8));
            }
        }
    }
    return result;
}
#endif

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
    auto options=::tyloi::TensorOptions().device(::tyloi::Device(::tyloi::kCUDA, model_.primary_device)).dtype(::tyloi::kInt64);
    auto text=::tyloi::cuda::tensor(state.text_row_ids,options);
    auto routing=dense.reshape({-1,1}).to(::tyloi::Device(::tyloi::kCUDA, model_.primary_device), ::tyloi::kBool);
    state.image_mask=dense; state.text_rows=std::move(text); state.routing_mask=std::move(routing);
}

void State::begin_forward(bool capture_raw_hidden) {
    shared->capture_dspark_targets = capture_raw_hidden;
    shared->dspark_target_hiddens.clear();
    if (capture_raw_hidden)
        shared->dspark_target_hiddens.resize(model_.config.dspark_target_layer_ids.size());
}

void State::reset(std::int64_t next_batch) {
#ifdef TYLOINFER_NATIVE_CUDA
    set_image_mask({});
#endif
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
        if (state.local_kv.defined()) result.push_back(state.local_kv);
        if (state.compressed_kv.defined()) result.push_back(state.compressed_kv);
        if (state.index_k.defined()) result.push_back(state.index_k);
        if (state.partial_kv.defined()) result.push_back(state.partial_kv);
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
#ifdef TYLOINFER_NATIVE_CUDA
        const auto copy=[&](const ::tyloi::Tensor& value) -> ::tyloi::Tensor {
            if(!value.defined())return {};
            auto result=tyloi::ops::clone(value);
            state.bytes+=session_tensor_bytes(result);return result;
        };
#else
        const auto copy=[&](const DenseTensor& value) -> ::tyloi::Tensor {
            if(!value.defined())return {};
            auto result=std::make_shared<DenseTensor>(value.clone());
            state.bytes+=session_tensor_bytes(*result);return result;
        };
#endif
        saved.local_kv=copy(source.local_kv);
        saved.compressed_length=source.compressed_length;saved.partial_length=source.partial_length;
        if(source.compressed_kv.defined())
#ifdef TYLOINFER_NATIVE_CUDA
            saved.compressed_kv=copy(tyloi::ops::slice(source.compressed_kv,1,0,source.compressed_length));
        if(source.index_k.defined())saved.index_k=copy(tyloi::ops::slice(source.index_k,1,0,source.compressed_length));
#else
            saved.compressed_kv=copy(source.compressed_kv.narrow(1,0,source.compressed_length));
        if(source.index_k.defined())saved.index_k=copy(source.index_k.narrow(1,0,source.compressed_length));
#endif
        saved.partial_kv=copy(source.partial_kv);saved.partial_score=copy(source.partial_score);
        validate_layer(block,saved,state.cache_pos);
        layers.push_back(std::move(saved));
    }
    return state;
}

#ifdef TYLOINFER_NATIVE_CUDA
class Candidate final : public tyloi::models::Restore {
public:
    Candidate(State& owner,const TextSessionState& snapshot)
        : owner_(owner), snapshot_(snapshot), layers_(owner.shared->layers) {
        const auto* saved=std::get_if<tyloi::models::deepseek_v41::Snapshot>(&snapshot_.payload);
        if(owner.speculative_start>=0 || owner.text_session_state_kind()!=TextSessionStateKind::DeepseekV41 ||
           !saved || saved->blocks.size()!=owner.model_.blocks.size() || snapshot_.cache_pos<=0 ||
           snapshot_.cache_pos>owner.model_.max_position_embeddings() ||
           std::size_t(snapshot_.cache_pos)!=snapshot_.tokens.size() || snapshot_.decode_position_delta!=0 ||
           !saved->engram || saved->engram->batch!=1 || saved->engram->position!=snapshot_.cache_pos)
            throw CudaSessionStateError("DeepSeek-V4.1 text session state is incompatible");
        hash_=std::make_unique<deepseek_v41_runtime::EngramHashState>(*owner.shared->engram_hash);
        hash_->restore(*saved->engram);
        for(std::size_t i=0;i<saved->blocks.size();++i) {
            const auto& b=*owner.model_.blocks[i];
            auto& layer=layers_.at(static_cast<std::size_t>(b.layer));
            const auto& entry=saved->blocks[i];
            if(layer.speculative_pending) throw CudaSessionStateError("DeepSeek-V4.1 speculative state is pending");
            validate_layer(b,entry,snapshot_.cache_pos);
            const tyloi::Device device(tyloi::DeviceType::cuda,b.cuda_device);
            for(const auto* tensor:{&entry.local_kv,&entry.compressed_kv,&entry.index_k,&entry.partial_kv,&entry.partial_score})
                if(tensor->defined() && tensor->device()!=device)
                    throw CudaSessionStateError("DeepSeek-V4.1 session device is incompatible");
            add(growth_,device,restore_tensor_bytes(layer.local_kv,entry.local_kv));
            if(b.kv_source()) {
                const auto capacity=std::max<std::int64_t>(1,b.max_context/b.ratio);
                if(entry.compressed_length>capacity) throw CudaSessionStateError("DeepSeek-V4.1 session capacity is incompatible");
                add(growth_,device,restore_prefix_bytes(layer.compressed_kv,entry.compressed_kv,1,capacity));
                add(growth_,device,restore_prefix_bytes(layer.index_k,entry.index_k,1,capacity));
                if(b.ratio>1) {
                    add(growth_,device,restore_tensor_bytes(layer.partial_kv,entry.partial_kv));
                    add(growth_,device,restore_tensor_bytes(layer.partial_score,entry.partial_score));
                }
            }
        }
    }
    std::vector<std::pair<tyloi::Device,std::size_t>> growth() const override { return growth_; }
    void stage() override {
        const auto& saved=std::get<tyloi::models::deepseek_v41::Snapshot>(snapshot_.payload);
        for(std::size_t i=0;i<saved.blocks.size();++i) {
            const auto& b=*owner_.model_.blocks[i];
            MfqCudaGuard guard(b.cuda_device);
            auto& layer=layers_.at(static_cast<std::size_t>(b.layer));
            const auto& entry=saved.blocks[i];
            copy(layer.local_kv,entry.local_kv);
            if(b.kv_source()) {
                const auto capacity=std::max<std::int64_t>(1,b.max_context/b.ratio);
                prefix(layer.compressed_kv,entry.compressed_kv,capacity);
                prefix(layer.index_k,entry.index_k,capacity);
                if(b.ratio>1) { copy(layer.partial_kv,entry.partial_kv); copy(layer.partial_score,entry.partial_score); }
                else { layer.partial_kv={}; layer.partial_score={}; }
            } else { layer.compressed_kv={}; layer.index_k={}; layer.partial_kv={}; layer.partial_score={}; }
            layer.position=snapshot_.cache_pos; layer.compressed_length=entry.compressed_length;
            layer.partial_length=entry.partial_length; layer.speculative_pending=false;
            layer.speculative_position=-1; layer.prefilled_global_end=-1; layer.replay_local_window=false;
            layer.speculative_local_slots={}; layer.speculative_slot_ids={}; layer.speculative_partial_kv={};
            layer.speculative_partial_score={}; layer.speculative_projected_kv={}; layer.speculative_projected_score={};
            layer.current_ids={};
        }
    }
    void publish() noexcept override {
        auto& s=*owner_.shared;
        s.layers.swap(layers_); s.engram_hash.swap(hash_);
        std::swap(s.compressed_kv,compressed_kv_); std::swap(s.index_k,index_k_);
        std::swap(s.topk,topk_); std::swap(s.candidate_blocks,candidate_blocks_);
        std::swap(s.previous_pre,previous_pre_); std::swap(s.engram_hashes,engram_hashes_);
        s.dspark_target_hiddens.swap(hiddens_); s.capture_dspark_targets=false;
        std::swap(s.image_mask,image_mask_); std::swap(s.text_rows,text_rows_);
        std::swap(s.routing_mask,routing_mask_); s.text_row_ids.swap(text_row_ids_);
        s.compressed_length=0; s.ratio=0; s.cache_position=-1; s.batch=0; s.tokens=0; s.next_layer=0;
        owner_.cache_pos=snapshot_.cache_pos; owner_.decode_position_delta=0; owner_.batch=1;
        owner_.speculative_start=-1; owner_.speculative_confirmed=0; owner_.speculative_suffix_forward=false;
    }
private:
    static void copy(tyloi::Tensor& target,const tyloi::Tensor& source) {
        if(restore_tensor_bytes(target,source)) target=tyloi::ops::empty(source.dense().sizes(),source.options());
        tyloi::ops::copy_into(target,source);
    }
    static void prefix(tyloi::Tensor& target,const tyloi::Tensor& source,std::int64_t capacity) {
        if(restore_prefix_bytes(target,source,1,capacity)) {
            auto shape=source.dense().sizes().vec(); shape[1]=capacity;
            target=tyloi::ops::empty(shape,source.options());
        }
        auto view=tyloi::ops::slice(target,1,0,source.size(1));
        tyloi::ops::copy_into(view,source);
    }
    State& owner_;
    TextSessionState snapshot_;
    std::vector<deepseek_v41_runtime::LayerState> layers_;
    std::unique_ptr<deepseek_v41_runtime::EngramHashState> hash_;
    Growth growth_;
    DenseTensor compressed_kv_,index_k_,topk_,candidate_blocks_,previous_pre_,image_mask_,text_rows_,routing_mask_;
    deepseek_v41_runtime::EngramHashBatch engram_hashes_;
    std::vector<DenseTensor> hiddens_;
    std::vector<std::int64_t> text_row_ids_;
};

std::unique_ptr<tyloi::models::Restore> State::prepare_restore(const TextSessionState& snapshot) {
    return std::make_unique<Candidate>(*this,snapshot);
}
std::vector<std::pair<tyloi::Device,std::size_t>> State::restore_growth(const TextSessionState& snapshot) const {
    return Candidate(const_cast<State&>(*this),snapshot).growth();
}
#else
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
        auto local_kv=restore_session_tensor(target.local_kv,saved.local_kv);
        target.local_kv=local_kv.defined()?local_kv.dense():DenseTensor{};
        if(block.kv_source()) {
            target.compressed_kv=restore_session_prefix_tensor(target.compressed_kv,saved.compressed_kv,1,
                                          std::max<int64_t>(1,block.max_context/block.ratio)).dense();
            target.index_k=restore_session_prefix_tensor(target.index_k,saved.index_k,1,
                                          std::max<int64_t>(1,block.max_context/block.ratio)).dense();
            if(block.ratio>1) {
                auto partial_kv=restore_session_tensor(target.partial_kv,saved.partial_kv);
                target.partial_kv=partial_kv.defined()?partial_kv.dense():DenseTensor{};
                auto partial_score=restore_session_tensor(target.partial_score,saved.partial_score);
                target.partial_score=partial_score.defined()?partial_score.dense():DenseTensor{};
            }
        }
        target.position=state.cache_pos;target.compressed_length=saved.compressed_length;
        target.partial_length=saved.partial_length;
    }
    *shared->engram_hash=std::move(hash);
    set_image_mask({});
    cache_pos=state.cache_pos;
}
#endif

} // namespace tyloi::cuda::deepseek_v41
