#include "engine/models/qwen4_exp/state.h"
#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/qwen4_exp/snapshot.h"
#include "engine/models/qwen4_exp/predictor.h"
#include "engine/models/session/state.h"

#include <algorithm>
#include <stdexcept>

namespace tyloi::cuda::qwen4_exp {

tyloi::models::SnapshotSize snapshot_size(const State& state,const tyloi::models::Mtp* mtp) {
    const auto& model = state.model_;
    std::size_t bytes=session_tensor_bytes(state.positions);
    const auto count_block=[&](const Qwen4Block& block,const LayerState& layer,int64_t count) {
        if(count<=0)return;
        if(block.qsa)bytes+=block.qsa->session_bytes(*layer.qsa,count);
        if(block.gdn) {
            bytes+=session_tensor_bytes(layer.gdn.conv);
            bytes+=session_tensor_bytes(layer.gdn.recurrent);
        }
        if(block.ple)bytes+=session_tensor_bytes(block.ple->conv_state(layer.ple));
    };
    for(std::size_t i=0;i<model.blocks.size();++i)count_block(*model.blocks[i],*state.layers[i],state.cache_pos);
    if(const auto* predictor=dynamic_cast<const Predictor*>(mtp)) {
        for(std::size_t i=0;i<predictor->layers.size();++i) {
            const auto count=std::min(state.cache_pos-1,predictor->lengths[i]);
            count_block(*predictor->layers[i],*predictor->layer_states[i],count);
            if(count>0)bytes+=std::size_t(count)*3*predictor->positions[i].element_size();
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
    for (std::size_t i=0;i<model_.blocks.size();++i) {
        const auto& block=*model_.blocks[i];
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
    for (std::size_t i=0; i<model_.blocks.size(); ++i)
        for (auto* tensor : model_.blocks[i]->graph_warmup_state(*layers[i]))
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
    for (std::size_t i=0; i<model_.blocks.size(); ++i)
        model_.blocks[i]->commit_speculative(*layers[i]);
    speculative_prefix_retention = false;
    speculative_start = -1;
    speculative_confirmed = 0;
}

TextSessionStateKind State::text_session_state_kind() const {
    return model_.execution && !model_.execution->offload && !model_.blocks.empty()
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
    for (std::size_t i=0; i<model_.blocks.size(); ++i)
        snapshot.blocks.push_back(capture_block_session(*model_.blocks[i], *layers[i], cache_pos, state.bytes));
    snapshot.positions = std::make_shared<DenseTensor>(positions.clone());
    state.bytes += session_tensor_bytes(*snapshot.positions);
    state.payload = std::move(snapshot);
    return state;
}

void State::restore_text_session_state(const TextSessionState& state) {
    const auto* snapshot = std::get_if<tyloi::models::qwen4_exp::Snapshot>(&state.payload);
    const auto* positions = snapshot ? dynamic_cast<const DenseTensor*>(snapshot->positions.get()) : nullptr;
    TYLOINFER_RUNTIME_CHECK(text_session_state_kind() == TextSessionStateKind::Qwen4 && snapshot &&
                                snapshot->blocks.size() == model_.blocks.size() && state.cache_pos > 0 &&
                                std::size_t(state.cache_pos) == state.tokens.size() &&
                                positions && positions->defined() &&
                                positions->size(-1) == state.cache_pos,
                            "incompatible Qwen4 text session state");
    reset(1);
    for (std::size_t i = 0; i < snapshot->blocks.size(); ++i)
        restore_block_session(*model_.blocks[i], *layers[i], snapshot->blocks[i], state.cache_pos);
    this->positions = positions->clone();
    cache_pos = state.cache_pos;
}

} // namespace tyloi::cuda::qwen4_exp
