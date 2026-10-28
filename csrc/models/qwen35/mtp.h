#pragma once

#include "models/qwen35/predictor.h"
#include "engine/executor/run.h"
#include "models/qwen35/mtp_state.h"
#include "engine/executor/mtp.h"

#include <functional>
#include <memory>

namespace mfq::cuda::qwen35 {

class State;
class Mtp;

void restore_session(State&, Mtp*, const TextSessionState&, const std::function<void()>&);

class Mtp final : public MtpModule {
public:
    Mtp(Predictor model, QuantLinear embedding, RopeCache rope,
        std::unique_ptr<Executor> executor, Collectives& copies,
        int device, const int64_t& kv_capacity, Trace* const& trace, const int& trace_scope,
        bool preallocate);

    void reset(int64_t batch = 1) override;
    mfq_tensor_backend::Tensor forward(
        mfq_tensor_backend::Tensor hidden, mfq_tensor_backend::Tensor next_ids) override;
    MtpStep step_positioned(mfq_tensor_backend::Tensor hidden,
        mfq_tensor_backend::Tensor next_ids, mfq_tensor_backend::Tensor positions) override;

    int64_t cache_position() const noexcept override { return state_.position(); }
    void trim_cache_to(int64_t position) override;
    bool supports_session_state() const noexcept override { return true; }
    MtpSessionState capture_session_state(int64_t target_cache_position,
        const mfq_tensor_backend::Tensor& last_target_hidden) const override;

    bool teacher_forced_prompt_prime() const noexcept override { return true; }
    bool target_bootstrap_decode() const noexcept override { return false; }
    bool preserve_output_dtype() const noexcept override { return false; }
    bool retains_partial_target_prefix() const noexcept override { return true; }

private:
    friend void restore_session(State&, Mtp*, const TextSessionState&,
        const std::function<void()>&);

    mfq_tensor_backend::Tensor evaluate(
        mfq_tensor_backend::Tensor hidden, mfq_tensor_backend::Tensor next_ids,
        mfq_tensor_backend::Tensor positions);

    Predictor model_;
    QuantLinear embedding_;
    RopeCache rope_;
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
