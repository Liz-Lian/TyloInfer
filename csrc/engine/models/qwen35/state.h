#pragma once

#include "engine/models/session/snapshot.h"
#include "engine/models/state.h"
#include "engine/state/kv_view.h"
#include "engine/state/recurrent.h"
#include "engine/backends/cuda/kernels/mfq_cuda_kv_quantization.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/ops/linear.h"
#include "engine/state/media.h"
#else
#include "engine/backends/cuda/ops/attention/linear.h"
#endif

#include <cstdint>
#include <variant>
#include <vector>

namespace tyloi::cuda::qwen35 {

struct Model;

// Linear attention keeps the projected rows until a speculative pass has
// either been committed or replayed from its confirmed prefix.
struct LinearState {
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::ops::Linear::State attention;
#else
    Recurrent mixer;
    LinearInputs projected;
#endif
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

class State
#ifdef TYLOINFER_NATIVE_CUDA
    : public tyloi::models::State
#endif
{
public:
    struct Restore {
        std::vector<LayerState> layers;
        std::int64_t cache_pos = 0;
        std::int64_t decode_position_delta = 0;
    };

    /** Create request state from model geometry and the recurrent layout policy. */
    State(const Model& model, const bool& transposed);
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;

#ifdef TYLOINFER_NATIVE_CUDA
    /** Check that this request state was created for the supplied model. */
    bool belongs_to(const Model& model) const noexcept { return model_ == &model; }
    /** Plan new persistent and speculative backing for a forward step. */
    std::vector<std::pair<tyloi::Device, std::size_t>> growth(
        std::int64_t batch, std::int64_t tokens, const tyloi::models::ForwardOptions& options,
        std::int64_t confirmed = 0) const override;
    /** Reset only reuses or releases existing backing. */
    std::vector<std::pair<tyloi::Device, std::size_t>> reset_growth(std::int64_t batch) const override;
    /** Replay the accepted part of a pending verification window. */
    void rollback_speculative(std::int64_t accepted) override;
    /** Report the actual verification transaction state. */
    bool speculating() const override { return speculative_start >= 0; }
    /** Return the per-layer recurrent verification batch counters. */
    std::optional<tyloi::engine::Metrics> metrics() const override;
    /** Report that replay retains the accepted verification prefix. */
    bool retains_partial_prefix(std::int64_t) const noexcept override { return true; }
    /** Validate and retain a snapshot for independent staged restoration. */
    std::unique_ptr<tyloi::models::Restore> prepare_restore(const TextSessionState&) override;
#endif

    void preallocate();
    /** Reset request state for a batch, retaining compatible storage. */
    void reset(std::int64_t new_batch);
    std::int64_t position() const noexcept { return cache_pos; }
    void advance(std::int64_t tokens) noexcept { cache_pos += tokens; }
    void set_position_delta(std::int64_t delta) noexcept { decode_position_delta = delta; }
    /** Commit a completed speculative verification window. */
    void commit_speculative();
    /** Rewind recurrent checkpoints to the accepted verification position. */
    std::int64_t rewind_suffix(std::int64_t accepted_suffix);
    /** Abort a pending verification and restore its starting position. */
    void abort_speculative() noexcept;

    /** Return stable recurrent tensor members for decode graph warmup. */
    std::vector<::tyloi::Tensor*> graph_state();
    /** Return owning handles for retained request and checkpoint storage. */
    std::vector<::tyloi::Tensor> storage() const;

    TextSessionStateKind text_session_state_kind() const;
    bool supports_text_session_state() const;
    bool supports_paged_text_session_state() const;
    /** Capture a confirmed text session in the existing wire format. */
    TextSessionState capture_text_session_state(
        const std::vector<std::int64_t>& tokens) const;
    /** Validate a snapshot and prepare an independent restore candidate. */
    Restore prepare(const TextSessionState& state) const;
    /** Copy a validated snapshot into the candidate before draining device work. */
    void stage(Restore& restore, const TextSessionState& state) const;
    /** Publish a drained candidate while retaining replaced layers in it. */
    void publish(Restore&& restore) noexcept;

    const std::vector<Layout>& layout() const noexcept { return layout_; }

    std::vector<LayerState> layers;
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::cache::Media images;
#endif
#ifndef TYLOINFER_NATIVE_CUDA
    std::unique_ptr<tyloi::models::MtpState> predictor;
#endif
    std::int64_t cache_pos = 0;
    std::int64_t decode_position_delta = 0;
    std::int64_t speculative_start = -1;
    std::int64_t speculative_confirmed = 0;
    std::int64_t batch = 0;

private:
    std::vector<Layout> layout_;
    std::int64_t max_position_embeddings_;
#ifdef TYLOINFER_NATIVE_CUDA
    const Model* model_;
    bool transposed_;
#else
    const bool* transposed_;
#endif
};

} // namespace tyloi::cuda::qwen35
