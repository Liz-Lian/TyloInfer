#pragma once

#include "engine/models/session/snapshot.h"
#include "engine/models/state.h"
#include <cstdint>
#include <vector>

namespace tyloi::cuda::deepseek_v4 {
struct Model;

struct PoolState {
    int64_t ratio = 0, head_dim = 0, cache_quant_mode = 0, capacity = 0;
    bool overlap = false;
    ::tyloi::Tensor state_kv, state_gate, previous_kv, previous_gate, pool;
    /** Allocate or reuse fixed compressor storage and clear its committed contents. */
    void reset(int64_t batch, int64_t max_positions);
};

struct LayerState { ::tyloi::Tensor local_cache; PoolState compressor, indexer_compressor; };

/** Request-owned DeepSeek V4 cache; its model must remain alive at the same address. */
class State : public tyloi::models::State {
public:
    /** Capture the model identity and per-layer cache layout. */
    explicit State(const Model&);
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;
    /** Check whether this request state was created by the given model. */
    bool belongs_to(const Model& model) const noexcept { return model_ == &model; }
    /** Allocate or reuse the batch caches and reset their contents and position. */
    void reset(int64_t batch) override;
    /** Return the committed cache position. */
    int64_t position() const noexcept override { return cache_pos; }
    /** Advance the committed position by completed tokens. */
    void advance(int64_t tokens) noexcept override { cache_pos += tokens; }
    /** Set the semantic position offset. */
    void set_position_delta(int64_t delta) noexcept override { decode_position_delta = delta; }
    /** Fixed cache allocation is performed by reset. */
    void preallocate() override {}
    /** Estimate per-device bytes newly allocated by resetting to the given batch. */
    std::vector<std::pair<tyloi::Device, std::size_t>> reset_growth(std::int64_t batch) const override;
    /** Report incremental allocation required by a forward step. */
    std::vector<std::pair<tyloi::Device, std::size_t>> growth(
        std::int64_t tokens, const tyloi::models::ForwardOptions& options,
        std::int64_t confirmed = 0) const override;
    /** Return owning handles for the current state storage. */
    std::vector<::tyloi::Tensor> storage() const override;
    /** Return recurrent tensors that require graph warmup restoration. */
    std::vector<::tyloi::Tensor*> graph_state() override;
    TextSessionStateKind text_session_state_kind() const;
    /** Report support for text session state. */
    bool supports_text_session_state() const override;
    /** Report support for paged text session state. */
    bool supports_paged_text_session_state() const override { return false; }
    /** Capture a text session snapshot for the committed tokens. */
    TextSessionState capture_text_session_state(const std::vector<int64_t>&) const override;
    /** Estimate per-device backing allocated when restoring the snapshot. */
    std::vector<std::pair<tyloi::Device, std::size_t>> restore_growth(
        const TextSessionState& snapshot) const override;
    /** Restore a compatible text session snapshot. */
    void restore_text_session_state(const TextSessionState&) override;

    std::vector<LayerState> layers;
    int64_t cache_pos = 0, decode_position_delta = 0, batch = 0;

private:
    const Model* model_;
    struct Layout { int device; int64_t head_dim, max_positions; };
    std::vector<Layout> layout_;
};
} // namespace tyloi::cuda::deepseek_v4
