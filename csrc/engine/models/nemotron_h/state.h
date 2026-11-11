#pragma once

#include "engine/state/recurrent.h"
#include "engine/state/kv_view.h"
#include "engine/models/state.h"
#include "engine/models/session/snapshot.h"

#include <cstdint>
#include <variant>
#include <vector>

namespace tyloi::cuda::nemotron_h {

struct Model;

struct MambaState {
    Recurrent mixer;
    ::tyloi::Tensor speculative_xbc, speculative_dt;
};

using LayerState = std::variant<std::monostate, MambaState, tyloi::state::KvView>;

/** Request-owned Nemotron-H state; its creating model must outlive it at a stable address. */
class State : public tyloi::models::State {
public:
    /** Capture the creating model identity and layer cache layouts. */
    explicit State(const Model&);
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;

    /** Return whether the state was created by the given model. */
    bool belongs_to(const Model& model) const noexcept { return model_ == &model; }
    /** Reset cache positions and incompatible storage for a batch. */
    void reset(std::int64_t batch) override;
    /** Return the committed cache position. */
    std::int64_t position() const noexcept override { return cache_pos; }
    /** Advance the committed cache position. */
    void advance(std::int64_t tokens) noexcept override { cache_pos += tokens; }
    /** Set the semantic decode position offset. */
    void set_position_delta(std::int64_t delta) noexcept override { decode_position_delta = delta; }
    /** Preallocate KV caches when configured. */
    void preallocate() override;
    /** Commit a speculative verification pass. */
    void commit_speculative() override;
    /** Roll back a speculative pass and replay the retained prefix. */
    void rollback_speculative(std::int64_t accepted) override;
    /** Conservatively estimate state and speculative replay allocations per device. */
    std::vector<std::pair<tyloi::Device, std::size_t>> growth(
        std::int64_t tokens, const tyloi::models::ForwardOptions& options,
        std::int64_t confirmed = 0) const override;
    /** Return owning handles for current cache and speculative storage. */
    std::vector<::tyloi::Tensor> storage() const override;
    /** Borrow writable warmup state until the next call, reset, restore, or destruction. */
    std::vector<::tyloi::Tensor*> graph_state() override;
    TextSessionStateKind text_session_state_kind() const;
    /** Report text session support for this layer layout. */
    bool supports_text_session_state() const override;
    /** Report whether all layers support paged text sessions. */
    bool supports_paged_text_session_state() const override;
    /** Capture a text snapshot for the committed tokens. */
    TextSessionState capture_text_session_state(const std::vector<std::int64_t>&) const override;
    /** Estimate allocations required to restore a compatible snapshot. */
    std::vector<std::pair<tyloi::Device, std::size_t>> restore_growth(const TextSessionState&) const override;
    /** Restore a compatible text session snapshot. */
    void restore_text_session_state(const TextSessionState&) override;

    struct MambaLayout {
        std::int64_t heads, head_dim, state_size, groups, conv_kernel;
    };
    struct FullLayout {
        std::int64_t heads, head_dim, capacity;
        bool sliding;
        ::tyloi::ScalarType dtype;
        tyloi::state::KvQuantization quantization;
    };
    struct Layout {
        int device;
        std::variant<std::monostate, MambaLayout, FullLayout> mixer;
    };
    const std::vector<Layout>& layout() const noexcept { return layout_; }

    std::vector<LayerState> layers;
    std::int64_t cache_pos = 0, decode_position_delta = 0, batch = 0;
    std::int64_t speculative_start = -1, speculative_confirmed = 0;

private:
    const Model* model_;
    std::vector<Layout> layout_;
#ifdef TYLOINFER_NATIVE_CUDA
    std::vector<::tyloi::Tensor> graph_views_;
#endif
};

} // namespace tyloi::cuda::nemotron_h
