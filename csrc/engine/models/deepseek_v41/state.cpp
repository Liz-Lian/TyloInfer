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
        const auto& entry=state.shared->layers[static_cast<std::size_t>(item->layer)];
        const auto& layer=entry.attention;
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
using Growth = std::vector<std::pair<tyloi::Device, std::size_t>>;
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
}

std::vector<std::pair<tyloi::Device, std::size_t>> State::reset_growth(std::int64_t next_batch) const {
    if(next_batch<=0) throw std::invalid_argument("DeepSeek-V4.1 reset batch must be positive");
    Growth result;
    for(const auto& item:model_.blocks) {
        const auto& b=*item;
        const auto& layer=shared->layers.at(static_cast<std::size_t>(b.layer)).attention;
        const tyloi::Device device(tyloi::DeviceType::cuda,b.cuda_device);
        add(result,device,b.attention->reset_growth(layer,next_batch));
    }
    return result;
}
std::vector<std::pair<tyloi::Device, std::size_t>> State::suffix_growth(std::int64_t tokens) const {
    if(batch!=1 || tokens<=0 || speculative_start>=0)
        throw std::invalid_argument("DeepSeek-V4.1 invalid speculative allocation window");
    Growth result;
    for(const auto& item:model_.blocks) {
        const auto& b=*item;
        const auto& layer=shared->layers.at(static_cast<std::size_t>(b.layer)).attention;
        const auto device=tyloi::Device(tyloi::DeviceType::cuda,b.cuda_device);
        add(result,device,b.attention->suffix_growth(layer,batch,tokens));
    }
    return result;
}

std::vector<std::pair<tyloi::Device, std::size_t>> State::growth(
    std::int64_t, std::int64_t tokens,const tyloi::models::ForwardOptions&,std::int64_t confirmed) const {
    if(batch<=0 || tokens<=0 || confirmed<0 || confirmed>=tokens)
        throw std::invalid_argument("DeepSeek-V4.1 invalid forward allocation window");
    Growth result=confirmed>0 ? suffix_growth(tokens) : Growth{};
    if(confirmed>0 || speculative_start>=0) {
        for(const auto& item:model_.blocks) {
            const auto& b=*item;
            const auto device=tyloi::Device(tyloi::DeviceType::cuda,b.cuda_device);
            const auto& layer=shared->layers.at(static_cast<std::size_t>(b.layer)).attention;
            add(result,device,b.attention->growth(layer,batch,tokens,true));
        }
    }
    return result;
}

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
    set_image_mask({});
    cache_pos = decode_position_delta = 0;
    speculative_start = -1;
    speculative_confirmed = 0;
    speculative_suffix_forward = false;
    batch = next_batch;
    for (auto& block : model_.blocks) {
        block->attention->reset(shared->layers.at(static_cast<std::size_t>(block->layer)).attention, next_batch);
        if (block->layer == 0) shared->reset_session(next_batch);
    }
}

std::vector<::tyloi::Tensor> State::storage() const {
    std::vector<::tyloi::Tensor> result;
    for (const auto& block : model_.blocks) {
        const auto& state = shared->layers.at(static_cast<std::size_t>(block->layer));
        auto tensors = block->attention->storage(state.attention);
        result.insert(result.end(), tensors.begin(), tensors.end());
        if (state.current_ids.defined()) result.push_back(state.current_ids);
    }
    const auto& attention = shared->attention;
    if (attention.compressed_kv.defined()) result.push_back(attention.compressed_kv);
    if (attention.index_k.defined()) result.push_back(attention.index_k);
    if (attention.topk.defined()) result.push_back(attention.topk);
    if (attention.candidate_blocks.defined()) result.push_back(attention.candidate_blocks);
    for (const auto* tensor : {&shared->ced_source, &shared->attention_input,
                               &shared->attention_positions, &shared->attention_lengths})
        if (tensor->defined()) result.emplace_back(tensor->dense());
    if (shared->previous_pre.defined()) result.push_back(shared->previous_pre);
    if (shared->image_mask.defined()) result.push_back(shared->image_mask);
    if (shared->text_rows.defined()) result.push_back(shared->text_rows);
    if (shared->routing_mask.defined()) result.push_back(shared->routing_mask);
    for (const auto& hidden : shared->dspark_target_hiddens)
        if (hidden.defined()) result.push_back(hidden);
    for (const auto& engram : shared->engram_states) {
        if (!engram) continue;
        auto buffers = engram->storage();
        result.insert(result.end(),buffers.begin(),buffers.end());
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
            block->attention->begin_speculative(shared->layers.at(static_cast<std::size_t>(block->layer)).attention, draft_tokens);
            ++begun;
        }
    } catch (...) {
        for (std::size_t i = 0; i < begun; ++i) {
            try {
                auto& block = *model_.blocks[i];
                block.attention->commit_speculative(shared->layers.at(static_cast<std::size_t>(block.layer)).attention);
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
        block->attention->commit_speculative(shared->layers.at(static_cast<std::size_t>(block->layer)).attention);
    }
    if (shared->engram_hash) shared->engram_hash->commit_speculative();
    speculative_start = -1;
    speculative_confirmed = 0;
}

void State::rollback_speculative(std::int64_t accepted) {
    using tyloi::models::require_model;
    require_model(speculative_start >= 0 && accepted >= 0, "no speculative transaction");
    const auto keep = speculative_start + speculative_confirmed + accepted;
    for (auto& block : model_.blocks) {
        block->attention->rollback_speculative(shared->layers.at(static_cast<std::size_t>(block->layer)).attention, keep);
    }
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
        const auto& entry=shared->layers.at(static_cast<std::size_t>(block.layer));
        const auto& source=entry.attention;
        if(source.speculative_pending || source.position!=cache_pos)
            throw std::runtime_error("DeepSeek-V4.1 session attention state is not committed");
        tyloi::models::deepseek_v41::BlockSnapshot saved;
        const auto copy=[&](const ::tyloi::Tensor& value) -> ::tyloi::Tensor {
            if(!value.defined())return {};
            auto result=tyloi::ops::clone(value);
            state.bytes+=session_tensor_bytes(result);return result;
        };
        saved.local_kv=copy(source.local_kv);
        saved.compressed_length=source.compressed_length;saved.partial_length=source.partial_length;
        if(source.compressed_kv.defined())
            saved.compressed_kv=copy(tyloi::ops::slice(source.compressed_kv,1,0,source.compressed_length));
        if(source.index_k.defined())saved.index_k=copy(tyloi::ops::slice(source.index_k,1,0,source.compressed_length));
        saved.partial_kv=copy(source.partial_kv);saved.partial_score=copy(source.partial_score);
        validate_layer(block,saved,state.cache_pos);
        layers.push_back(std::move(saved));
    }
    return state;
}

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
        for(const auto& engram:owner.shared->engram_states)if(engram) {
            auto buffers=engram->storage();
            engram_storage_.insert(engram_storage_.end(),buffers.begin(),buffers.end());
        }
        for(std::size_t i=0;i<saved->blocks.size();++i) {
            const auto& b=*owner.model_.blocks[i];
            auto& layer=layers_.at(static_cast<std::size_t>(b.layer)).attention;
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
            auto& current=layers_.at(static_cast<std::size_t>(b.layer));
            auto& layer=current.attention;
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
            current.current_ids={};
        }
    }
    void publish() noexcept override {
        auto& s=*owner_.shared;
        s.layers.swap(layers_); s.engram_hash.swap(hash_);
        for(auto& engram:s.engram_states)if(engram)engram->clear();
        std::swap(s.attention,attention_);
        std::swap(s.ced_source,ced_source_); std::swap(s.attention_input,attention_input_);
        std::swap(s.attention_positions,attention_positions_); std::swap(s.attention_lengths,attention_lengths_);
        std::swap(s.previous_pre,previous_pre_); std::swap(s.engram_hashes,engram_hashes_);
        s.dspark_target_hiddens.swap(hiddens_); s.capture_dspark_targets=false;
        std::swap(s.image_mask,image_mask_); std::swap(s.text_rows,text_rows_);
        std::swap(s.routing_mask,routing_mask_); s.text_row_ids.swap(text_row_ids_);
        s.cache_position=-1; s.batch=0; s.tokens=0; s.next_layer=0;
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
    std::vector<tyloi::Tensor> engram_storage_;
    tyloi::ops::Csa::Shared attention_;
    tyloi::Tensor ced_source_{DenseTensor{}}, attention_input_{DenseTensor{}},
        attention_positions_{DenseTensor{}}, attention_lengths_{DenseTensor{}};
    DenseTensor previous_pre_,image_mask_,text_rows_,routing_mask_;
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

} // namespace tyloi::cuda::deepseek_v41
