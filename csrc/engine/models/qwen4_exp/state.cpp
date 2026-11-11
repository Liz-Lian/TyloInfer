#include "engine/models/qwen4_exp/state.h"
#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/qwen4_exp/snapshot.h"
#include "engine/models/qwen4_exp/predictor.h"
#include "engine/models/session/state.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tyloi::cuda::qwen4_exp {

#ifdef TYLOINFER_NATIVE_CUDA
namespace {
using Growth = std::vector<std::pair<tyloi::Device, std::size_t>>;
std::size_t multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("Qwen4 allocation planning overflow");
    return a * b;
}
void add(Growth& result, tyloi::Device device, std::size_t bytes) {
    if (!bytes) return;
    auto it = std::find_if(result.begin(), result.end(), [device](const auto& entry) { return entry.first == device; });
    if (it == result.end()) result.emplace_back(device, bytes);
    else {
        if (it->second > std::numeric_limits<std::size_t>::max() - bytes)
            throw std::overflow_error("Qwen4 allocation planning overflow");
        it->second += bytes;
    }
}
}

std::vector<std::pair<tyloi::Device, std::size_t>> State::reset_growth(std::int64_t next_batch) const {
    if (next_batch <= 0) throw std::invalid_argument("Qwen4 reset batch must be positive");
    return {};
}

std::vector<std::pair<tyloi::Device, std::size_t>> State::suffix_growth(std::int64_t tokens) const {
    if (tokens <= 0 || speculative_start >= 0 || cache_pos <= 0 || batch != 1 ||
        tokens > model_.max_position_embeddings() - cache_pos)
        throw std::invalid_argument("Qwen4 invalid speculative suffix");
    Growth result;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto& weights = *model_.layers[i];
        const auto& layer = *layers[i];
        const auto device = tyloi::Device(tyloi::DeviceType::cuda, weights.cuda_device);
        if (weights.gdn) {
            if (!layer.gdn.conv.defined() || !layer.gdn.recurrent.defined() ||
                layer.gdn.saved_conv.defined() || layer.gdn.saved_state.defined())
                throw std::invalid_argument("Qwen4 recurrent checkpoint is unavailable");
            add(result, device, session_tensor_bytes(layer.gdn.conv));
            add(result, device, session_tensor_bytes(layer.gdn.recurrent));
        }
        if (weights.ple) {
            if (!layer.ple.conv.defined() || layer.ple.saved)
                throw std::invalid_argument("Qwen4 PLE checkpoint is unavailable");
            add(result, device, session_tensor_bytes(layer.ple.conv));
        }
    }
    return result;
}

std::vector<std::pair<tyloi::Device, std::size_t>> State::growth(
        std::int64_t tokens, const tyloi::models::ForwardOptions& options, std::int64_t confirmed) const {
    if (batch <= 0 || tokens <= 0 || confirmed < 0 || confirmed >= tokens || cache_pos < 0 ||
        tokens > model_.max_position_embeddings() - cache_pos)
        throw std::invalid_argument("Qwen4 invalid forward allocation window");
    Growth result;
    const auto primary = tyloi::Device(tyloi::DeviceType::cuda, options.primary);
    add(result, primary, multiply(multiply(multiply(std::size_t(batch), 3),
        std::size_t(cache_pos + tokens)), sizeof(std::int64_t)));
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto& weights = *model_.layers[i];
        const auto& layer = *layers[i];
        const auto device = tyloi::Device(tyloi::DeviceType::cuda, weights.cuda_device);
        if (weights.qsa) {
            add(result, device, multiply(weights.qsa->growth_bytes(*layer.qsa, tokens), std::size_t(batch)));
            if (layer.qsa->graph_keys.defined() && !weights.qsa->managed_memory(*layer.qsa))
                add(result, device, multiply(weights.qsa->session_bytes(*layer.qsa,cache_pos+tokens),2));
        }
        if (weights.gdn) {
            add(result, device, multiply(weights.gdn->request_cache_growth_bytes(layer.gdn), std::size_t(batch)));
            if (confirmed > 0) {
                const auto copies = std::size_t(tokens <= 8 ? tokens - confirmed : 1);
                add(result, device, multiply(copies, session_tensor_bytes(layer.gdn.conv)));
                add(result, device, multiply(copies, session_tensor_bytes(layer.gdn.recurrent)));
            }
        }
        if (weights.ple) {
            const auto bytes = layer.ple.conv.defined() ? session_tensor_bytes(layer.ple.conv) :
                multiply(multiply(multiply(multiply(std::size_t(batch), std::size_t(weights.config.ple_kernel - 1)),
                    std::size_t(weights.config.ngram)), std::size_t(weights.config.hidden)),
                    multiply(std::size_t(weights.config.streams), sizeof(float)));
            if (!layer.ple.conv.defined()) add(result, device, bytes);
            if (confirmed > 0) add(result, device, multiply(bytes, std::size_t(tokens <= 8 ? tokens - confirmed : 1)));
        }
    }
    return result;
}
#endif

LayerState::LayerState(const Layer& layer) {
    if (layer.qsa)
        qsa = std::make_unique<Qsa::State>(*layer.qsa);
}

void LayerState::reset(const Layer& layer) {
    if (layer.gdn)
        layer.gdn->reset(gdn, retain_graph_storage);
    if (layer.qsa)
        layer.qsa->reset(*qsa);
    if (layer.ple)
        layer.ple->reset(ple, retain_graph_storage);
}

void LayerState::commit(const Layer& layer) {
    if (layer.gdn)
        layer.gdn->commit(gdn);
    if (layer.ple)
        layer.ple->commit(ple);
}

void LayerState::rollback(const Layer& layer, std::int64_t keep) {
    if (layer.gdn)
        layer.gdn->rollback(gdn, keep);
    if (layer.qsa)
        layer.qsa->truncate(*qsa, keep);
    if (layer.ple)
        layer.ple->rollback(ple, keep);
}

std::vector<::tyloi::Tensor*> LayerState::graph_state(const Layer& layer) {
    std::vector<::tyloi::Tensor*> result;
    if (layer.gdn)
        for (auto* value : layer.gdn->graph_state(gdn))
            if (value->defined()) result.push_back(value);
    if (layer.ple)
        for (auto* value : layer.ple->graph_state(ple))
            if (value->defined()) result.push_back(value);
    return result;
}

tyloi::models::qwen4_exp::LayerSnapshot capture_layer_session(
        const Layer& layer, const LayerState& state, std::int64_t position, std::size_t& bytes) {
    tyloi::models::qwen4_exp::LayerSnapshot snapshot;
    const auto save = [&](const DenseTensor& tensor) {
        TYLOINFER_RUNTIME_CHECK(tensor.defined(), "Qwen4 recurrent session state is unavailable");
        auto saved = std::make_shared<DenseTensor>(tensor.clone());
        bytes += session_tensor_bytes(*saved);
        return saved;
    };
    if (layer.qsa)
        layer.qsa->capture_session(*state.qsa, snapshot, position, bytes);
    if (layer.gdn) {
        snapshot.convolution = save(state.gdn.conv.dense());
        snapshot.recurrent = save(state.gdn.recurrent.dense());
    }
    if (layer.ple) {
        snapshot.ple_convolution = save(layer.ple->conv_state(state.ple).dense());
        const auto& context = layer.ple->session_context(state.ple);
        snapshot.ple_batch = context.batch;
        snapshot.ple_tokens = context.tokens;
        bytes += snapshot.ple_tokens.size() * sizeof(std::int64_t);
    }
    return snapshot;
}

#ifndef TYLOINFER_NATIVE_CUDA
void restore_layer_session(
        const Layer& layer, LayerState& state,
        const tyloi::models::qwen4_exp::LayerSnapshot& snapshot, std::int64_t position) {
    state.reset(layer);
    if (layer.qsa)
        layer.qsa->restore_session(*state.qsa, snapshot, position);
    if (layer.gdn) {
        const auto* convolution = snapshot.convolution.representation<DenseTensor>();
        const auto* recurrent = snapshot.recurrent.representation<DenseTensor>();
        TYLOINFER_RUNTIME_CHECK(convolution && recurrent, "missing Qwen4 GDN session state");
        layer.gdn->restore_session(state.gdn, *convolution, *recurrent, state.retain_graph_storage);
    }
    if (layer.ple) {
        const auto* convolution = snapshot.ple_convolution.representation<DenseTensor>();
        TYLOINFER_RUNTIME_CHECK(convolution, "missing Qwen4 PLE session state");
        layer.ple->restore_session(state.ple, *convolution,
                                   {snapshot.ple_batch, snapshot.ple_tokens}, state.retain_graph_storage);
    }
}

#else
void validate_layer_session(const Layer& layer, const LayerState& current,
        const tyloi::models::qwen4_exp::LayerSnapshot& saved, std::int64_t position) {
    const auto device=tyloi::Device(tyloi::DeviceType::cuda,layer.cuda_device);
    const auto check=[&](const tyloi::Tensor& value, std::initializer_list<std::int64_t> shape) {
        if (!value.defined() || !value.representation<DenseTensor>() || value.device()!=device ||
            value.scalar_type()!=tyloi::kFloat32 || value.dim()!=std::int64_t(shape.size())) return false;
        std::int64_t i=0;
        for (auto dimension:shape) if(value.size(i++)!=dimension) return false;
        return true;
    };
    if (layer.qsa) {
        TYLOINFER_RUNTIME_CHECK(current.qsa, "Qwen4 QSA state is unavailable");
        layer.qsa->validate_session(*current.qsa,saved,position,device);
    } else TYLOINFER_RUNTIME_CHECK(!saved.keys && !saved.values && !saved.index,
        "unexpected Qwen4 QSA snapshot");
    const auto& c=layer.config;
    if (layer.gdn) {
        TYLOINFER_RUNTIME_CHECK(!current.gdn.saved_conv.defined() &&
            check(saved.convolution,{1,c.kernel-1,(2*c.key_heads+c.value_heads)*c.linear_width}) &&
            check(saved.recurrent,{1,c.value_heads,c.linear_width,c.linear_width}),
            "incompatible Qwen4 GDN session state");
    } else TYLOINFER_RUNTIME_CHECK(!saved.convolution && !saved.recurrent,
        "unexpected Qwen4 GDN snapshot");
    if (layer.ple) {
        TYLOINFER_RUNTIME_CHECK(!current.ple.saved &&
            check(saved.ple_convolution,{1,(c.ple_kernel-1)*c.ngram,c.hidden*c.streams}) &&
            saved.ple_batch==1 && saved.ple_tokens.size()==std::size_t(c.ngram-1),
            "incompatible Qwen4 PLE session state");
    } else TYLOINFER_RUNTIME_CHECK(!saved.ple_convolution && saved.ple_batch==0 && saved.ple_tokens.empty(),
        "unexpected Qwen4 PLE snapshot");
}

std::size_t layer_restore_growth(const Layer& layer, const LayerState& current,
        const tyloi::models::qwen4_exp::LayerSnapshot& saved, std::int64_t position) {
    validate_layer_session(layer,current,saved,position);
    Growth result;
    const auto device=tyloi::Device(tyloi::DeviceType::cuda,layer.cuda_device);
    if (layer.qsa) add(result,device,layer.qsa->fresh_session_growth(*current.qsa,position));
    if (layer.gdn) { add(result,device,session_tensor_bytes(saved.convolution)); add(result,device,session_tensor_bytes(saved.recurrent)); }
    if (layer.ple) add(result,device,session_tensor_bytes(saved.ple_convolution));
    return result.empty()?0:result.front().second;
}

void stage_layer_session(const Layer& layer, LayerState& candidate,
        const tyloi::models::qwen4_exp::LayerSnapshot& saved, std::int64_t position) {
    if (layer.qsa) layer.qsa->stage_session(*candidate.qsa,saved,position);
    const auto copy=[](tyloi::Tensor& target,const tyloi::Tensor& source) {
        target=tyloi::ops::empty(source.dense().sizes(),source.options());
        tyloi::ops::copy_into(target,source);
    };
    if (layer.gdn) { copy(candidate.gdn.conv,saved.convolution); copy(candidate.gdn.recurrent,saved.recurrent); }
    if (layer.ple) {
        copy(candidate.ple.conv,saved.ple_convolution);
        layer.ple->restore_graph_context(candidate.ple,{saved.ple_batch,saved.ple_tokens});
    }
}
#endif

tyloi::models::SnapshotSize snapshot_size(const State& state,const tyloi::models::Mtp* mtp) {
    const auto& model = state.model_;
    std::size_t bytes=session_tensor_bytes(state.positions);
    const auto count_block=[&](const Layer& block,const LayerState& layer,int64_t count) {
        if(count<=0)return;
        if(block.qsa)bytes+=block.qsa->session_bytes(*layer.qsa,count);
        if(block.gdn) {
            bytes+=session_tensor_bytes(layer.gdn.conv);
            bytes+=session_tensor_bytes(layer.gdn.recurrent);
        }
        if(block.ple)bytes+=session_tensor_bytes(block.ple->conv_state(layer.ple));
    };
    for(std::size_t i=0;i<model.layers.size();++i)count_block(*model.layers[i],*state.layers[i],state.cache_pos);
    if(const auto* predictor=dynamic_cast<const Predictor*>(mtp); predictor && state.predictor && state.predictor->belongs_to(*predictor)) {
        const auto& request=static_cast<const PredictorState&>(*state.predictor);
        for(std::size_t i=0;i<predictor->layers.size();++i) {
            const auto count=std::min(state.cache_pos-1,request.lengths[i]);
            count_block(*predictor->layers[i],*request.layer_states[i],count);
            if(count>0)bytes+=std::size_t(count)*3*request.positions[i].element_size();
        }
        bytes+=std::size_t(model.config.hidden)*model.config.streams*sizeof(float);
    }
    // Transposed graph KV can temporarily require a contiguous copy of the checkpoint.
    return {bytes, bytes * 2};
}

std::vector<::tyloi::Tensor> State::storage() const {
    std::vector<::tyloi::Tensor> result;
    const auto add=[&](const ::tyloi::Tensor& value) { if(value.defined())result.push_back(value); };
    add(positions);
    for (std::size_t i=0;i<model_.layers.size();++i) {
        const auto& block=*model_.layers[i];
        const auto& layer=*layers[i];
        if(block.gdn) {
            const auto& gdn=layer.gdn;
            // GDN reassigns the Dense representation in place. Capture an
            // independent view so a later reset cannot replace this backing.
            add(gdn.conv.dense());add(gdn.recurrent.dense());add(gdn.core_workspace.dense());
            add(gdn.saved_conv);add(gdn.saved_state);
            for(const auto& value:gdn.saved_prefix_conv)add(value);
            for(const auto& value:gdn.saved_prefix_state)add(value);
        }
        if(block.ple) {
            const auto& ple=layer.ple;
            add(ple.conv);add(ple.saved_conv);
            for(const auto& value:ple.saved_prefix_conv)add(value);
        }
        if(block.qsa) {
            const auto& qsa=*layer.qsa;
            add(qsa.keys.storage());add(qsa.values.storage());add(qsa.index.storage());
            add(qsa.graph_keys);add(qsa.graph_values);add(qsa.graph_pooled);add(qsa.graph_index);
            if(qsa.memory) {
                add(qsa.memory->keys);add(qsa.memory->values);add(qsa.memory->index);add(qsa.memory->pooled);
                const auto& hot=qsa.memory->hot;
                add(hot.keys);add(hot.values);add(hot.mapping);add(hot.slot_ids);add(hot.requested);
                add(hot.epochs);add(hot.queue);add(hot.control);add(hot.cold_keys);add(hot.cold_values);
                add(hot.cold_mapping);add(hot.free_cold);add(hot.free_gpu);add(hot.cold_control);
            }
        }
    }
    return result;
}

std::vector<tyloi::Tensor*> State::graph_state() {
    std::vector<tyloi::Tensor*> warmup_states;
    for (std::size_t i=0; i<model_.layers.size(); ++i)
        for (auto* tensor : layers[i]->graph_state(*model_.layers[i]))
            warmup_states.push_back(tensor);
    return warmup_states;
}

void State::begin_speculative_suffix(std::int64_t draft_tokens) {
    tyloi::models::require_model(model_.supports_suffix_speculation() && draft_tokens > 0 &&
                                     cache_pos + draft_tokens <= model_.max_position_embeddings(),
                                 "invalid speculative suffix");
#ifdef TYLOINFER_NATIVE_CUDA
    (void)suffix_growth(draft_tokens);
    for (std::size_t i=0; i<model_.layers.size(); ++i) {
        const auto& weights=*model_.layers[i];
        auto& layer=*layers[i];
        MfqCudaGuard guard(weights.cuda_device);
        if (weights.gdn) {
            layer.gdn.saved_conv=layer.gdn.conv.dense().clone();
            layer.gdn.saved_state=layer.gdn.recurrent.dense().clone();
        }
        if (weights.ple) {
            layer.ple.saved_context=weights.ple->session_context(layer.ple);
            layer.ple.saved_conv=layer.ple.conv.dense().clone();
            layer.ple.saved=true;
        }
    }
    speculative_prefix_retention=false;
#endif
    speculative_start = cache_pos;
    speculative_confirmed = 0;
}

void State::commit_speculative() {
    tyloi::models::require_model(speculative_start >= 0, "no speculative transaction");
    for (std::size_t i=0; i<model_.layers.size(); ++i)
        layers[i]->commit(*model_.layers[i]);
    speculative_prefix_retention = false;
    speculative_start = -1;
    speculative_confirmed = 0;
}

TextSessionStateKind State::text_session_state_kind() const {
    return model_.execution && !model_.execution->offload && !model_.layers.empty()
        ? TextSessionStateKind::Qwen4
        : TextSessionStateKind::Unsupported;
}

bool State::supports_text_session_state() const {
    return text_session_state_kind() != TextSessionStateKind::Unsupported;
}

bool State::supports_paged_text_session_state() const {
    return text_session_state_kind() == TextSessionStateKind::Qwen4;
}

TextSessionState State::capture_text_session_state(const std::vector<std::int64_t>& tokens) const {
    TYLOINFER_RUNTIME_CHECK(text_session_state_kind() == TextSessionStateKind::Qwen4 && batch == 1 &&
                                cache_pos > 0 && std::size_t(cache_pos) == tokens.size() &&
                                speculative_start < 0,
                            "Qwen4 text session boundary is unavailable");
    TextSessionState state;
    state.tokens = tokens;
    state.cache_pos = cache_pos;
    tyloi::models::qwen4_exp::Snapshot snapshot;
    for (std::size_t i=0; i<model_.layers.size(); ++i)
        snapshot.layers.push_back(capture_layer_session(*model_.layers[i], *layers[i], cache_pos, state.bytes));
#ifdef TYLOINFER_NATIVE_CUDA
    snapshot.positions = tyloi::ops::clone(positions);
#else
    snapshot.positions = std::make_shared<DenseTensor>(positions.clone());
#endif
    state.bytes += session_tensor_bytes(snapshot.positions);
    state.payload = std::move(snapshot);
    return state;
}

#ifdef TYLOINFER_NATIVE_CUDA
class Candidate final : public tyloi::models::Restore {
public:
    Candidate(State& owner, const TextSessionState& saved) : owner_(owner), saved_(saved) {
        const auto* snapshot=std::get_if<tyloi::models::qwen4_exp::Snapshot>(&saved_.payload);
        if(owner_.speculative_start>=0 || owner_.text_session_state_kind()!=TextSessionStateKind::Qwen4 ||
           !snapshot || snapshot->layers.size()!=owner_.model_.layers.size() ||
           owner_.layers.size()!=owner_.model_.layers.size() || saved_.cache_pos<=0 ||
           saved_.cache_pos>owner_.model_.max_position_embeddings() || saved_.decode_position_delta!=0 ||
           std::size_t(saved_.cache_pos)!=saved_.tokens.size())
            throw CudaSessionStateError("incompatible Qwen4 text session state");
        const auto& positions=snapshot->positions;
        const auto primary=tyloi::Device(tyloi::DeviceType::cuda,
            owner_.model_.execution->config.layer_placement.primary_device());
        if(!positions.defined() || !positions.representation<DenseTensor>() || positions.device()!=primary ||
           positions.scalar_type()!=tyloi::kInt64 || positions.size(-1)!=saved_.cache_pos ||
           !(positions.dim()==1 || (positions.dim()==2 && positions.size(0)==3) ||
             (positions.dim()==3 && positions.size(0)==3 && positions.size(1)==1)))
            throw CudaSessionStateError("incompatible Qwen4 session positions");
        add(growth_,primary,session_tensor_bytes(positions));
        for(std::size_t i=0;i<snapshot->layers.size();++i) {
            const auto& block=*owner_.model_.layers[i];
            add(growth_,tyloi::Device(tyloi::DeviceType::cuda,block.cuda_device),
                layer_restore_growth(block,*owner_.layers[i],snapshot->layers[i],saved_.cache_pos));
        }
    }
    std::vector<std::pair<tyloi::Device,std::size_t>> growth() const override { return growth_; }
    void stage() override {
        const auto& snapshot=std::get<tyloi::models::qwen4_exp::Snapshot>(saved_.payload);
        positions_=tyloi::ops::empty(snapshot.positions.dense().sizes(),snapshot.positions.options());
        tyloi::ops::copy_into(positions_,snapshot.positions);
        layers_.reserve(snapshot.layers.size());
        for(std::size_t i=0;i<snapshot.layers.size();++i) {
            const auto& block=*owner_.model_.layers[i];
            MfqCudaGuard guard(block.cuda_device);
            layers_.push_back(std::make_unique<LayerState>(block));
            stage_layer_session(block,*layers_.back(),snapshot.layers[i],saved_.cache_pos);
        }
    }
    void publish() noexcept override {
        owner_.retire_graphs();
        owner_.layers.swap(layers_);
        std::swap(owner_.positions,positions_);
        owner_.cache_pos=saved_.cache_pos; owner_.decode_position_delta=0; owner_.batch=1;
        owner_.speculative_start=-1; owner_.speculative_confirmed=0;
        owner_.speculative_suffix_forward=false; owner_.speculative_prefix_retention=false;
        owner_.decode_reserve_until=0;
    }
private:
    State& owner_;
    TextSessionState saved_;
    Growth growth_;
    tyloi::Tensor positions_;
    std::vector<std::unique_ptr<LayerState>> layers_;
};

std::unique_ptr<tyloi::models::Restore> State::prepare_restore(const TextSessionState& saved) {
    return std::make_unique<Candidate>(*this,saved);
}

#else
void State::restore_text_session_state(const TextSessionState& state) {
    const auto* snapshot = std::get_if<tyloi::models::qwen4_exp::Snapshot>(&state.payload);
    const auto* positions = snapshot ? snapshot->positions.representation<DenseTensor>() : nullptr;
    TYLOINFER_RUNTIME_CHECK(text_session_state_kind() == TextSessionStateKind::Qwen4 && snapshot &&
                                snapshot->layers.size() == model_.layers.size() && state.cache_pos > 0 &&
                                std::size_t(state.cache_pos) == state.tokens.size() &&
                                positions && positions->defined() &&
                                positions->size(-1) == state.cache_pos,
                            "incompatible Qwen4 text session state");
    reset(1);
    for (std::size_t i = 0; i < snapshot->layers.size(); ++i)
        restore_layer_session(*model_.layers[i], *layers[i], snapshot->layers[i], state.cache_pos);
    this->positions = positions->clone();
    cache_pos = state.cache_pos;
}
#endif

} // namespace tyloi::cuda::qwen4_exp
