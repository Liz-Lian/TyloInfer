#pragma once

#include "engine/models/qwen35/predictor.h"
#include "engine/models/session/snapshot.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/executor.h"
#endif
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/ops/rope/rope.h"
#endif
#include "engine/models/qwen35/mtp_state.h"
#include "engine/models/common/mtp.h"

#ifndef TYLOINFER_NATIVE_CUDA
#include <functional>
#endif
#include <memory>

namespace tyloi::cuda::qwen35 {

#ifndef TYLOINFER_NATIVE_CUDA
class State;
class Mtp;
void restore_session(State&, Mtp*, const TextSessionState&, const std::function<void()>&);
#endif

class Mtp final : public tyloi::models::Mtp {
public:
#ifdef TYLOINFER_NATIVE_CUDA
    /** Own predictor weights and placement; execution options arrive with each step. */
    Mtp(Predictor model, ::tyloi::Tensor embedding, tyloi::ops::RopeTable rope,
        int device);
#else
    Mtp(Predictor model, ::tyloi::Tensor embedding, RopeCache rope,
        std::unique_ptr<Executor> executor, Collectives& copies,
        int device, const int64_t& kv_capacity, Trace* const& trace, const int& trace_scope,
        bool preallocate);
#endif

    std::unique_ptr<tyloi::models::MtpState> make_state() const override;
    void reset(tyloi::models::MtpState& state, int64_t batch = 1) const override;
    /** Advance the predictor using the request's current capacity and trace policy. */
    tyloi::models::MtpStep step(tyloi::models::MtpState& state, const ::tyloi::Tensor& hidden,
        const ::tyloi::Tensor& next_ids, const ::tyloi::Tensor& positions = {},
        const tyloi::models::ForwardOptions& options = {}) const override;
#ifdef TYLOINFER_NATIVE_CUDA
    /** Plan predictor KV backing for the actual step geometry and capacity. */
    std::vector<std::pair<tyloi::Device, std::size_t>> growth(
        const tyloi::models::MtpState& state, const tyloi::models::Mtp::Input& input) const override;
    /** Validate and retain a predictor session restore candidate. */
    std::unique_ptr<tyloi::models::Restore> prepare_restore(
        tyloi::models::MtpState& state, const MtpSessionState& snapshot) const override;
#endif

    void trim_cache_to(tyloi::models::MtpState& state, int64_t position) const override;
    bool supports_session_state() const noexcept override { return true; }
    MtpSessionState capture_session_state(const tyloi::models::MtpState& state, int64_t target_cache_position,
        const ::tyloi::Tensor& last_target_hidden) const override;

    bool teacher_forced_prompt_prime() const noexcept override { return true; }
    bool target_bootstrap_decode() const noexcept override { return false; }
    bool preserve_output_dtype() const noexcept override { return false; }
    bool retains_partial_target_prefix() const noexcept override { return true; }

private:
#ifndef TYLOINFER_NATIVE_CUDA
    friend void restore_session(State&, Mtp*, const TextSessionState&,
        const std::function<void()>&);
#endif

#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor evaluate(MtpState& state, const ::tyloi::Tensor& hidden,
        const ::tyloi::Tensor& next_ids, const ::tyloi::Tensor& positions,
        const tyloi::models::ForwardOptions& options) const;
#else
    ::tyloi::DenseTensor evaluate(MtpState& state, ::tyloi::DenseTensor hidden,
        ::tyloi::DenseTensor next_ids, ::tyloi::DenseTensor positions) const;
#endif

    Predictor model_;
    ::tyloi::Tensor embedding_;
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::ops::RopeTable rope_;
#else
    RopeCache rope_;
#endif
#ifndef TYLOINFER_NATIVE_CUDA
    std::unique_ptr<Executor> executor_;
    Collectives& copies_;
    const int64_t& kv_capacity_;
    Trace* const& trace_;
    const int& trace_scope_;
#endif
    int device_;
#ifndef TYLOINFER_NATIVE_CUDA
    bool preallocate_;
#endif
};

} // namespace tyloi::cuda::qwen35
