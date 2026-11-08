#include "engine/models/qwen4_exp/state.h"
#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/qwen4_exp/snapshot.h"
#include "engine/models/qwen4_exp/predictor.h"
#include "engine/models/session/state.h"

#include <algorithm>
#include <stdexcept>

namespace tyloi::cuda::qwen4_exp {

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

void restore_layer_session(
        const Layer& layer, LayerState& state,
        const tyloi::models::qwen4_exp::LayerSnapshot& snapshot, std::int64_t position) {
    state.reset(layer);
    if (layer.qsa)
        layer.qsa->restore_session(*state.qsa, snapshot, position);
    if (layer.gdn) {
        const auto* convolution = dynamic_cast<const DenseTensor*>(snapshot.convolution.get());
        const auto* recurrent = dynamic_cast<const DenseTensor*>(snapshot.recurrent.get());
        TYLOINFER_RUNTIME_CHECK(convolution && recurrent, "missing Qwen4 GDN session state");
        layer.gdn->restore_session(state.gdn, *convolution, *recurrent, state.retain_graph_storage);
    }
    if (layer.ple) {
        const auto* convolution = dynamic_cast<const DenseTensor*>(snapshot.ple_convolution.get());
        TYLOINFER_RUNTIME_CHECK(convolution, "missing Qwen4 PLE session state");
        layer.ple->restore_session(state.ple, *convolution,
                                   {snapshot.ple_batch, snapshot.ple_tokens}, state.retain_graph_storage);
    }
}

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
    snapshot.positions = std::make_shared<DenseTensor>(positions.clone());
    state.bytes += session_tensor_bytes(*snapshot.positions);
    state.payload = std::move(snapshot);
    return state;
}

void State::restore_text_session_state(const TextSessionState& state) {
    const auto* snapshot = std::get_if<tyloi::models::qwen4_exp::Snapshot>(&state.payload);
    const auto* positions = snapshot ? dynamic_cast<const DenseTensor*>(snapshot->positions.get()) : nullptr;
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

} // namespace tyloi::cuda::qwen4_exp
