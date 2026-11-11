#pragma once

#include "engine/models/session/snapshot.h"
#include "engine/models/state.h"
#include <cstdint>
#include <map>
#include <vector>

namespace tyloi::cuda::glm_dsa {
using DenseTensor = ::tyloi::DenseTensor;
struct Model;

struct LayerState { DenseTensor kv_cache, index_cache; };
struct Selection { DenseTensor topk_indices; int64_t dense_prefix_rows = 0; };

class State : public tyloi::models::State {
public:
    explicit State(const Model&);
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;
    /** Return whether the state was created by this model. */
    bool belongs_to(const Model& model) const noexcept { return model_ == &model; }
    /** Reset cache positions and incompatible storage for a batch. */
    void reset(int64_t batch) override;
    /** Return the committed cache position. */
    int64_t position() const noexcept override { return cache_pos; }
    /** Advance the committed cache position. */
    void advance(int64_t tokens) noexcept override { cache_pos += tokens; }
    /** Set the semantic position offset used for decode. */
    void set_position_delta(int64_t delta) noexcept override { decode_position_delta = delta; }
    /** This model allocates its caches on demand. */
    void preallocate() override {}
    /** Estimate missing fixed-capacity MLA and index cache bytes per device. */
    std::vector<std::pair<tyloi::Device, std::size_t>> growth(
        std::int64_t batch, std::int64_t tokens, const tyloi::models::ForwardOptions& options,
        std::int64_t confirmed = 0) const override;
    /** Return owning handles for the current state storage. */
    std::vector<::tyloi::Tensor> storage() const override;
    /** Return recurrent tensors that require graph warmup restoration. */
    std::vector<::tyloi::Tensor*> graph_state() override { return {}; }
    TextSessionStateKind text_session_state_kind() const;
    /** Return whether this state supports text session snapshots. */
    bool supports_text_session_state() const override;
    /** Report that paged text session snapshots are unsupported. */
    bool supports_paged_text_session_state() const override { return false; }
    /** Capture a text session snapshot for the committed tokens. */
    TextSessionState capture_text_session_state(const std::vector<int64_t>&) const override;
    /** Estimate allocations needed to restore a compatible text session snapshot. */
    std::vector<std::pair<tyloi::Device, std::size_t>> restore_growth(
        const TextSessionState& snapshot) const override;
    /** Restore a compatible text session snapshot. */
    void restore_text_session_state(const TextSessionState&) override;

    std::vector<LayerState> layers;
    std::map<int, Selection> selections;
    int64_t cache_pos = 0, decode_position_delta = 0, batch = 0;

private:
    const Model* model_;
    struct Layout { int device; bool full_indexer; int64_t capacity, index_width; };
    std::vector<Layout> layout_;
};
} // namespace tyloi::cuda::glm_dsa
