#pragma once

#include "engine/models/session.h"
#include <cstdint>
#include <map>
#include <vector>

namespace tyloi::cuda::glm_dsa {
using DenseTensor = ::tyloi::DenseTensor;
struct Model;

struct LayerState { DenseTensor kv_cache, index_cache; };
struct Selection { DenseTensor topk_indices; int64_t dense_prefix_rows = 0; };

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
    std::vector<DenseTensor> storage() const;
    std::vector<DenseTensor*> graph_state() { return {}; }
    TextSessionStateKind text_session_state_kind() const;
    bool supports_text_session_state() const;
    bool supports_paged_text_session_state() const { return false; }
    TextSessionState capture_text_session_state(const std::vector<int64_t>&) const;
    void restore_text_session_state(const TextSessionState&);

    std::vector<LayerState> layers;
    std::map<int, Selection> selections;
    int64_t cache_pos = 0, decode_position_delta = 0, batch = 0;

private:
    struct Layout { int device; bool full_indexer; };
    std::vector<Layout> layout_;
};
} // namespace tyloi::cuda::glm_dsa
