#pragma once

#include "engine/models/state.h"
#include "engine/state/kv_view.h"

#include <cstdint>
#include <vector>

namespace tyloi::cuda::gemma4 {

struct Model;

/** Request-owned Gemma4 cache and positions; its creating model must outlive it and stay at the same address. */
class State : public tyloi::models::State {
public:
    /** Capture the creating model's identity and cache layout. */
    explicit State(const Model&);
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;

    /** Return whether this state was created for the given model. */
    bool belongs_to(const Model& model) const noexcept { return model_ == &model; }
    /** Allocate missing KV caches for a single-item batch. */
    void preallocate() override;
    /** Reset positions and discard caches incompatible with the new batch. */
    void reset(std::int64_t batch) override;
    /** Return the committed cache position. */
    std::int64_t position() const noexcept override { return cache_pos; }
    /** Advance the committed cache position by completed tokens. */
    void advance(std::int64_t tokens) noexcept override { cache_pos += tokens; }
    /** Set the offset used to derive semantic decode positions. */
    void set_position_delta(std::int64_t delta) noexcept override { decode_position_delta = delta; }
    /** Estimate per-device bytes needed for missing KV caches. */
    std::vector<std::pair<tyloi::Device, std::size_t>> growth(
        std::int64_t tokens, const tyloi::models::ForwardOptions& options) const override;
    /** Return owning handles to the cache backing used by Graph capture. */
    std::vector<::tyloi::Tensor> storage() const override;
    /** Borrow writable warmup views until the next call, move assignment, or state destruction. */
    std::vector<tyloi::Tensor*> graph_state() override;

    /** Report whether text session restoration is supported. */
    bool supports_text_session_state() const override;
    /** Report whether paged text session restoration is supported. */
    bool supports_paged_text_session_state() const override;

    struct Layout {
        int device;
        std::int64_t heads, head_dim, capacity;
        bool sliding;
        tyloi::ScalarType dtype;
        tyloi::state::KvQuantization quantization;
    };
    /** Return the per-layer cache allocation layout. */
    const std::vector<Layout>& layout() const noexcept { return layout_; }

    std::vector<tyloi::state::KvView> layers;
    std::int64_t cache_pos = 0;
    std::int64_t decode_position_delta = 0;
    std::int64_t batch = 0;

private:
    const Model* model_;
    std::vector<Layout> layout_;
    std::vector<::tyloi::Tensor> graph_views_;
};

} // namespace tyloi::cuda::gemma4
