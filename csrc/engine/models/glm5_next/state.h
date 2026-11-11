#pragma once

#include "engine/models/common/attention_ops.h"
#include "engine/models/state.h"
#include "engine/models/session/snapshot.h"
#include <cstdint>
#include <variant>
#include <vector>

namespace tyloi::cuda::glm5_next {

struct Model;

struct KdaState {
    ::tyloi::Tensor conv, recurrent, rollback_conv, rollback_recurrent;
    /** Release recurrent state and any speculative checkpoint. */
    void reset() { conv = {}; recurrent = {}; commit(); }
    bool pending() const { return rollback_conv.defined(); }
    /** Retire the speculative checkpoint after its prefix is committed. */
    void commit() { rollback_conv = {}; rollback_recurrent = {}; }
    /** Restore the checkpoint and retire its rollback handles. */
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

/** Request-owned GLM5 state; the creating model must outlive it at a stable address. */
class State : public tyloi::models::State {
public:
    /** Capture the model identity and construct its layer states. */
    explicit State(const Model&);
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;
    /** Return whether this state was created by the given model. */
    bool belongs_to(const Model& model) const noexcept { return model_ == &model; }
    /** Reset the layer caches and position for a new batch. */
    void reset(int64_t batch) override;
    /** Return the committed cache position. */
    int64_t position() const noexcept override { return cache_pos; }
    /** Advance the committed cache position. */
    void advance(int64_t tokens) noexcept override { cache_pos += tokens; }
    /** Set the semantic decode position offset. */
    void set_position_delta(int64_t delta) noexcept override { decode_position_delta = delta; }
    /** Leave cache allocation to forward. */
    void preallocate() override {}
    /** Commit the pending speculative checkpoint. */
    void commit_speculative() override;
    /** Roll back a speculative suffix while retaining the accepted prefix. */
    void rollback_speculative(std::int64_t accepted) override;
    /** Estimate cache allocations, conservatively including KDA convolution backing and temporary input. */
    std::vector<std::pair<tyloi::Device, std::size_t>> growth(
        std::int64_t tokens, const tyloi::models::ForwardOptions& options,
        std::int64_t confirmed = 0) const override;
    /** Return the state storage handles available to Graph capture. */
    std::vector<::tyloi::Tensor> storage() const override;
    /** Return writable state handles available for Graph warmup. */
    std::vector<::tyloi::Tensor*> graph_state() override { return {}; }
    TextSessionStateKind text_session_state_kind() const { return TextSessionStateKind::Unsupported; }
    /** Report that text sessions are unsupported. */
    bool supports_text_session_state() const override { return false; }
    /** Report that paged text sessions are unsupported. */
    bool supports_paged_text_session_state() const override { return false; }
    /** Reject unsupported session capture. */
    TextSessionState capture_text_session_state(const std::vector<int64_t>&) const override;
    /** Reject unsupported session restoration. */
    void restore_text_session_state(const TextSessionState&) override;

    std::vector<LayerState> layers;
    int64_t cache_pos = 0, decode_position_delta = 0, batch = 0;
    int64_t speculative_start = -1, speculative_confirmed = 0;
private:
    const Model* model_;
    struct Layout { int device; std::int64_t heads, width, kernel; };
    std::vector<Layout> layout_;
};

} // namespace tyloi::cuda::glm5_next
