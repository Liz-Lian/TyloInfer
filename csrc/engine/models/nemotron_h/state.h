#pragma once

#include "engine/state/recurrent.h"
#include "engine/state/kv_cache.h"
#include "engine/models/session.h"

#include <cstdint>
#include <variant>
#include <vector>

namespace mfq::cuda::nemotron_h {

struct Model;

struct MambaState {
    Recurrent mixer;
    mfq_tensor_backend::Tensor speculative_xbc, speculative_dt;
};

using LayerState = std::variant<std::monostate, MambaState, KVCache>;

class State {
public:
    explicit State(const Model&);
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;

    void reset(std::int64_t batch);
    void preallocate();
    void commit_speculative();
    std::vector<mfq_tensor_backend::Tensor> storage() const;
    std::vector<mfq_tensor_backend::Tensor*> graph_state();
    TextSessionStateKind text_session_state_kind() const;
    bool supports_text_session_state() const;
    bool supports_paged_text_session_state() const;
    TextSessionState capture_text_session_state(const std::vector<std::int64_t>&) const;
    void restore_text_session_state(const TextSessionState&);

    struct MambaLayout {
        std::int64_t heads, head_dim, state_size, groups, conv_kernel;
    };
    struct FullLayout {
        std::int64_t heads, head_dim, capacity;
        bool sliding;
        mfq_tensor_backend::ScalarType dtype;
        KvQuantization quantization;
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
    std::vector<Layout> layout_;
};

} // namespace mfq::cuda::nemotron_h
