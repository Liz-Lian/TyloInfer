#pragma once

#include "models/qwen35/predictor.h"
#include "engine/executor/run.h"
#include "models/qwen35/mtp_state.h"
#include "models/mtp.h"

#include <memory>

namespace mfq::cuda::qwen35 {

class Mtp final : public MtpModule {
public:
    Mtp(Predictor model, std::unique_ptr<Executor> executor, Collectives& copies,
        int device, const int64_t& kv_capacity, Trace* const& trace, const int& trace_scope,
        bool preallocate);

    void reset(int64_t batch = 1) override;
    mfq_tensor_backend::Tensor forward(const MtpTarget& target,
        mfq_tensor_backend::Tensor hidden, mfq_tensor_backend::Tensor next_ids) override;
    MtpStep step_positioned(const MtpTarget& target, mfq_tensor_backend::Tensor hidden,
        mfq_tensor_backend::Tensor next_ids, mfq_tensor_backend::Tensor positions) override;

    int64_t cache_position() const noexcept override { return state_.position(); }
    void trim_cache_to(int64_t position) override;
    bool supports_session_state() const noexcept override { return true; }
    MtpSessionState capture_session_state(int64_t target_cache_position,
        const mfq_tensor_backend::Tensor& last_target_hidden) const override;
    void restore_session_state(const MtpSessionState& state) override;

    bool teacher_forced_prompt_prime() const noexcept override { return true; }
    bool target_bootstrap_decode() const noexcept override { return false; }
    bool preserve_output_dtype() const noexcept override { return false; }
    bool retains_partial_target_prefix() const noexcept override { return true; }

private:
    mfq_tensor_backend::Tensor evaluate(const MtpTarget& target,
        mfq_tensor_backend::Tensor hidden, mfq_tensor_backend::Tensor next_ids,
        mfq_tensor_backend::Tensor positions);

    Predictor model_;
    MtpState state_;
    std::vector<Task> tasks_;
    std::unique_ptr<Executor> executor_;
    Collectives& copies_;
    int device_;
    const int64_t& kv_capacity_;
    Trace* const& trace_;
    const int& trace_scope_;
};

} // namespace mfq::cuda::qwen35
