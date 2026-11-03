#pragma once

#include "engine/models/session.h"
#include <cstdint>
#include <vector>

namespace mfq::cuda::deepseek_v4 {
using Tensor = mfq_tensor_backend::Tensor;
struct Model;

struct PoolState {
    int64_t ratio = 0, head_dim = 0, cache_quant_mode = 0, capacity = 0;
    bool overlap = false;
    Tensor state_kv, state_gate, previous_kv, previous_gate, pool;
    void reset(int64_t batch, int64_t max_positions);
};

struct LayerState { Tensor local_cache; PoolState compressor, indexer_compressor; };

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
    std::vector<Tensor> storage() const;
    std::vector<Tensor*> graph_state() { return {}; }
    TextSessionStateKind text_session_state_kind() const;
    bool supports_text_session_state() const;
    bool supports_paged_text_session_state() const { return false; }
    TextSessionState capture_text_session_state(const std::vector<int64_t>&) const;
    void restore_text_session_state(const TextSessionState&);

    std::vector<LayerState> layers;
    int64_t cache_pos = 0, decode_position_delta = 0, batch = 0;

private:
    struct Layout { int device; int64_t head_dim, max_positions; };
    std::vector<Layout> layout_;
};
} // namespace mfq::cuda::deepseek_v4
