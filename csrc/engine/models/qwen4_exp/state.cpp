#include "engine/models/qwen4_exp/state.h"
#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/session.h"
#include "engine/models/qwen4_exp/snapshot.h"

#include <stdexcept>

namespace tyloi::cuda::qwen4_exp {

State::State(Model& model) : model_(model) {
    model_.active = this;
    batch = model.batch;
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
    model_.adapter_reset(next_batch);
    for (auto& block : model_.blocks)
        block->reset(next_batch);
}

std::vector<std::shared_ptr<const tyloi::Tensor>> State::storage() const {
    std::vector<std::shared_ptr<const tyloi::Tensor>> result;
    for (const auto& block : model_.blocks)
        for (auto* tensor : block->graph_warmup_state())
            result.push_back(std::make_shared<tyloi::DenseTensor>(*tensor));
    return result;
}

std::vector<tyloi::Tensor*> State::graph_state() {
    std::vector<tyloi::Tensor*> warmup_states;
    for (const auto& block : model_.blocks)
        for (auto* state : block->graph_warmup_state())
            warmup_states.push_back(state);
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
    for (auto& block : model_.blocks)
        block->commit_speculative();
    model_.adapter_commit_speculative();
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
    TYLOINFER_RUNTIME_CHECK(text_session_state_kind() == TextSessionStateKind::Qwen4 && model_.batch == 1 &&
                                cache_pos > 0 && std::size_t(cache_pos) == tokens.size() &&
                                speculative_start < 0,
                            "Qwen4 text session boundary is unavailable");
    TextSessionState state;
    state.tokens = tokens;
    state.cache_pos = cache_pos;
    tyloi::models::qwen4_exp::Snapshot snapshot;
    for (const auto& block : model_.blocks)
        snapshot.blocks.push_back(capture_block_session(*block, cache_pos, state.bytes));
    snapshot.positions = model_.positions.clone();
    state.bytes += session_tensor_bytes(snapshot.positions);
    state.payload = std::move(snapshot);
    return state;
}

void State::restore_text_session_state(const TextSessionState& state) {
    const auto* snapshot = std::get_if<tyloi::models::qwen4_exp::Snapshot>(&state.payload);
    TYLOINFER_RUNTIME_CHECK(text_session_state_kind() == TextSessionStateKind::Qwen4 && snapshot &&
                                snapshot->blocks.size() == model_.blocks.size() && state.cache_pos > 0 &&
                                std::size_t(state.cache_pos) == state.tokens.size() &&
                                snapshot->positions.defined() &&
                                snapshot->positions.size(-1) == state.cache_pos,
                            "incompatible Qwen4 text session state");
    reset(1);
    for (std::size_t i = 0; i < snapshot->blocks.size(); ++i)
        restore_block_session(*model_.blocks[i], snapshot->blocks[i], state.cache_pos);
    model_.positions = snapshot->positions.clone();
    cache_pos = state.cache_pos;
}

} // namespace tyloi::cuda::qwen4_exp
