#pragma once

#include "engine/models/common/session/state.h"
#include "engine/state/kv_cache.h"

#include <cstdint>
#include <vector>

namespace mfq::cuda::minicpmo45 {

struct Model;

class State {
public:
    struct Restore {
        std::vector<KVCache> layers;
        int64_t cache_pos = 0;
        int64_t decode_position_delta = 0;
    };

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
    TextSessionState capture_text_session_state(const std::vector<int64_t>& tokens) const;
    Restore prepare(const TextSessionState&) const;
    void stage(Restore&, const TextSessionState&) const;
    void publish(Restore&&) noexcept;

    std::vector<KVCache> layers;
    int64_t cache_pos = 0;
    int64_t decode_position_delta = 0;
    int64_t batch = 0;

private:
    struct Layout {
        int device;
        bool cpu;
        int64_t heads, head_dim, capacity;
    };
    std::vector<Layout> layout_;
    int64_t maximum_;
};

} // namespace mfq::cuda::minicpmo45
