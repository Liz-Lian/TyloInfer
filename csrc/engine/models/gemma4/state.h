#pragma once

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/models/state.h"
#include "engine/state/kv_view.h"
#else
#include "engine/models/session.h"
#include "engine/state/kv_cache.h"
#endif

#include <cstdint>
#include <vector>

namespace tyloi::cuda::gemma4 {

struct Model;

class State
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    : public tyloi::models::State
#endif
{
public:
    explicit State(const Model&);
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    void preallocate() override;
    void reset(std::int64_t batch) override;
    std::int64_t position() const noexcept override { return cache_pos; }
    void advance(std::int64_t tokens) noexcept override { cache_pos += tokens; }
    void set_position_delta(std::int64_t delta) noexcept override { decode_position_delta = delta; }
    std::vector<tyloi::Tensor> storage() const override;
    std::vector<tyloi::Tensor*> graph_state() override { return {}; }
#else
    void preallocate();
    void reset(std::int64_t batch);
    std::int64_t position() const noexcept { return cache_pos; }
    void advance(std::int64_t tokens) noexcept { cache_pos += tokens; }
    void set_position_delta(std::int64_t delta) noexcept { decode_position_delta = delta; }
    std::vector<::tyloi::Tensor> storage() const;
    std::vector<::tyloi::Tensor*> graph_state() { return {}; }
#endif

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    bool supports_text_session_state() const override;
    bool supports_paged_text_session_state() const override;
#else
    TextSessionStateKind text_session_state_kind() const;
    bool supports_text_session_state() const;
    bool supports_paged_text_session_state() const;
    TextSessionState capture_text_session_state(const std::vector<std::int64_t>&) const;
    void restore_text_session_state(const TextSessionState&);
#endif

    struct Layout {
        int device;
        std::int64_t heads, head_dim, capacity;
        bool sliding;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        tyloi::ScalarType dtype;
        tyloi::state::KvQuantization quantization;
#else
        ::tyloi::ScalarType dtype;
        KvQuantization quantization;
#endif
    };
    const std::vector<Layout>& layout() const noexcept { return layout_; }

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    std::vector<tyloi::state::KvView> layers;
#else
    std::vector<KVCache> layers;
#endif
    std::int64_t cache_pos = 0;
    std::int64_t decode_position_delta = 0;
    std::int64_t batch = 0;

private:
    std::vector<Layout> layout_;
};

} // namespace tyloi::cuda::gemma4
