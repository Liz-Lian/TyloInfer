#pragma once

#include "engine/models/session.h"
#include "engine/state/kv_cache.h"

#include <vector>

namespace mfq::cuda::minicpmo45::text {
struct Model;

class State {
public:
    explicit State(const Model&);
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;

    void preallocate();
    void reset(int64_t batch);
    std::vector<mfq_tensor_backend::Tensor> storage() const;
    std::vector<mfq_tensor_backend::Tensor*> graph_state() { return {}; }
    TextSessionStateKind text_session_state_kind() const;
    bool supports_text_session_state() const;
    bool supports_paged_text_session_state() const;
    TextSessionState capture_text_session_state(const std::vector<int64_t>&) const;
    void restore_text_session_state(const TextSessionState&);

    std::vector<KVCache> layers;
    int64_t cache_pos = 0, decode_position_delta = 0, batch = 0;

private:
    struct Layout {
        int device;
        bool cpu, sliding;
        int64_t heads, head_dim, capacity;
        KvQuantization quantization;
    };
    std::vector<Layout> layout_;
};
} // namespace mfq::cuda::minicpmo45::text
