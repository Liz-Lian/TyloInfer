#pragma once

#include "engine/models/session.h"
#include "engine/state/kv_cache.h"

#include <cstdint>
#include <vector>

namespace mfq::cuda::gemma4 {

struct Model;

class State {
public:
    explicit State(const Model&);
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;

    void preallocate();
    void reset(std::int64_t batch);
    std::vector<mfq_tensor_backend::Tensor> storage() const;
    std::vector<mfq_tensor_backend::Tensor*> graph_state() { return {}; }

    TextSessionStateKind text_session_state_kind() const;
    bool supports_text_session_state() const;
    bool supports_paged_text_session_state() const;
    TextSessionState capture_text_session_state(const std::vector<std::int64_t>&) const;
    void restore_text_session_state(const TextSessionState&);

    struct Layout {
        int device;
        std::int64_t heads, head_dim, capacity;
        bool sliding;
        mfq_tensor_backend::ScalarType dtype;
        KvQuantization quantization;
    };
    const std::vector<Layout>& layout() const noexcept { return layout_; }

    std::vector<KVCache> layers;
    std::int64_t cache_pos = 0;
    std::int64_t decode_position_delta = 0;
    std::int64_t batch = 0;

private:
    std::vector<Layout> layout_;
};

} // namespace mfq::cuda::gemma4
