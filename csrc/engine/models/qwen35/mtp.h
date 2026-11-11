#pragma once

#include "engine/models/qwen35/predictor.h"
#include "engine/models/session/snapshot.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/models/qwen35/mtp_state.h"
#include "engine/models/common/mtp.h"

#include <functional>
#include <memory>

namespace tyloi::cuda::qwen35 {

class State;
class Mtp;

void restore_session(State&, Mtp*, const TextSessionState&, const std::function<void()>&);

class Mtp final : public tyloi::models::Mtp {
public:
    Mtp(Predictor model, ::tyloi::Tensor embedding,
#ifdef TYLOINFER_NATIVE_CUDA
        tyloi::ops::RopeTable rope,
#else
        RopeCache rope,
#endif
        std::unique_ptr<Executor> executor, Collectives& copies,
        int device, const int64_t& kv_capacity, Trace* const& trace, const int& trace_scope,
        bool preallocate);

    std::unique_ptr<tyloi::models::MtpState> make_state() const override;
    void reset(tyloi::models::MtpState& state, int64_t batch = 1) const override;
    tyloi::models::MtpStep step(tyloi::models::MtpState& state, const ::tyloi::Tensor& hidden,
        const ::tyloi::Tensor& next_ids, const ::tyloi::Tensor& positions = {}) const override;

    void trim_cache_to(tyloi::models::MtpState& state, int64_t position) const override;
    bool supports_session_state() const noexcept override { return true; }
    MtpSessionState capture_session_state(const tyloi::models::MtpState& state, int64_t target_cache_position,
        const ::tyloi::Tensor& last_target_hidden) const override;

    bool teacher_forced_prompt_prime() const noexcept override { return true; }
    bool target_bootstrap_decode() const noexcept override { return false; }
    bool preserve_output_dtype() const noexcept override { return false; }
    bool retains_partial_target_prefix() const noexcept override { return true; }

private:
    friend void restore_session(State&, Mtp*, const TextSessionState&,
        const std::function<void()>&);

    ::tyloi::DenseTensor evaluate(
        MtpState& state, ::tyloi::DenseTensor hidden, ::tyloi::DenseTensor next_ids,
        ::tyloi::DenseTensor positions) const;

    Predictor model_;
    ::tyloi::Tensor embedding_;
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::ops::RopeTable rope_;
#else
    RopeCache rope_;
#endif
    std::unique_ptr<Executor> executor_;
    Collectives& copies_;
    int device_;
    const int64_t& kv_capacity_;
    Trace* const& trace_;
    const int& trace_scope_;
    bool preallocate_;
};

} // namespace tyloi::cuda::qwen35
