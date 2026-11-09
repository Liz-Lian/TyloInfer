#pragma once

#include "engine/models/session/snapshot.h"
#include "engine/models/state.h"
#include "engine/state/kv_view.h"
#include "engine/state/recurrent.h"
#include "engine/backends/cuda/kernels/mfq_cuda_kv_quantization.h"
#include "engine/backends/cuda/ops/attention/linear.h"

#include <cstdint>
#include <variant>
#include <vector>

namespace tyloi::cuda::qwen35 {

struct Model;

// Linear attention keeps the projected rows until a speculative pass has
// either been committed or replayed from its confirmed prefix.
struct LinearState {
    Recurrent mixer;
    LinearInputs projected;
    std::int64_t speculative_ffn_batches = 0;
    std::int64_t speculative_projection_batches = 0;
};

using LayerState = std::variant<tyloi::state::KvView, LinearState>;

struct FullLayout {
    std::int64_t kv_heads = 0;
    std::int64_t head_dim = 0;
    bool sliding = false;
    ::tyloi::ScalarType dtype = ::tyloi::kFloat16;
    tyloi::state::KvQuantization quantization;
    std::int64_t capacity = 0;
};

struct LinearLayout {
    std::int64_t key_heads = 0;
    std::int64_t key_dim = 0;
    std::int64_t value_heads = 0;
    std::int64_t value_dim = 0;
    std::int64_t conv_kernel = 0;
};

struct Layout {
    int device = 0;
    bool cpu = false;
    std::variant<FullLayout, LinearLayout> attention;
};

class State {
public:
    struct Restore {
        std::vector<LayerState> layers;
        std::int64_t cache_pos = 0;
        std::int64_t decode_position_delta = 0;
    };

    State(const Model& model, const bool& transposed);
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;

    void preallocate();
    void reset(std::int64_t new_batch);
    std::int64_t position() const noexcept { return cache_pos; }
    void advance(std::int64_t tokens) noexcept { cache_pos += tokens; }
    void set_position_delta(std::int64_t delta) noexcept { decode_position_delta = delta; }
    void commit_speculative();
    std::int64_t rewind(std::int64_t accepted_suffix);
    void abort_speculative() noexcept;

    std::vector<::tyloi::DenseTensor*> graph_state();
    std::vector<::tyloi::Tensor> storage() const;

    TextSessionStateKind text_session_state_kind() const;
    bool supports_text_session_state() const;
    bool supports_paged_text_session_state() const;
    TextSessionState capture_text_session_state(
        const std::vector<std::int64_t>& tokens) const;
    Restore prepare(const TextSessionState& state) const;
    void stage(Restore& restore, const TextSessionState& state) const;
    void publish(Restore&& restore) noexcept;

    const std::vector<Layout>& layout() const noexcept { return layout_; }

    std::vector<LayerState> layers;
    std::unique_ptr<tyloi::models::MtpState> predictor;
    std::int64_t cache_pos = 0;
    std::int64_t decode_position_delta = 0;
    std::int64_t speculative_start = -1;
    std::int64_t speculative_confirmed = 0;
    std::int64_t batch = 0;

private:
    std::vector<Layout> layout_;
    std::int64_t max_position_embeddings_;
    const bool* transposed_;
};

} // namespace tyloi::cuda::qwen35
