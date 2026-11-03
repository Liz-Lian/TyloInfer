#pragma once

#include "engine/models/common/attention_ops.h"
#include "engine/models/session.h"
#include <cstdint>
#include <variant>
#include <vector>

namespace mfq::cuda::glm5_next {

using Tensor = mfq_tensor_backend::Tensor;
struct Model;

struct KdaState {
    Tensor conv, recurrent, rollback_conv, rollback_recurrent;
    void reset() { conv = Tensor(); recurrent = Tensor(); commit(); }
    bool pending() const { return rollback_conv.defined(); }
    void commit() { rollback_conv = Tensor(); rollback_recurrent = Tensor(); }
    void rollback() {
        TYLOINFER_RUNTIME_CHECK(pending(), "GLM KDA has no speculative checkpoint");
        conv = rollback_conv;
        recurrent = rollback_recurrent;
        commit();
    }
};

struct MlaState {
    attention_ops::SequenceCache latent, index;
    MlaState(int64_t maximum, int64_t width, int64_t index_width)
        : latent(maximum, width), index(maximum, 2 * index_width) {}
    void reset() { latent.reset(); index.reset(); }
    int64_t position() const { return latent.position(); }
    void truncate(int64_t keep) {
        TYLOINFER_RUNTIME_CHECK(keep >= 0 && keep <= latent.position() && keep <= index.position(),
                               "invalid Flash-Next MLA cache truncation");
        latent.truncate(keep);
        index.truncate(keep);
    }
};

using LayerState = std::variant<KdaState, MlaState>;

class State {
public:
    explicit State(const Model&);
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;
    void reset(int64_t batch);
    int64_t position() const noexcept { return cache_pos; }
    void advance(int64_t tokens) noexcept { cache_pos += tokens; }
    void set_position_delta(int64_t delta) noexcept { decode_position_delta = delta; }
    void preallocate() {}
    void commit_speculative();
    std::vector<Tensor> storage() const { return {}; }
    std::vector<Tensor*> graph_state() { return {}; }
    TextSessionStateKind text_session_state_kind() const { return TextSessionStateKind::Unsupported; }
    bool supports_text_session_state() const { return false; }
    bool supports_paged_text_session_state() const { return false; }
    TextSessionState capture_text_session_state(const std::vector<int64_t>&) const;
    void restore_text_session_state(const TextSessionState&);

    std::vector<LayerState> layers;
    int64_t cache_pos = 0, decode_position_delta = 0, batch = 0;
    int64_t speculative_start = -1, speculative_confirmed = 0;
};

} // namespace mfq::cuda::glm5_next
