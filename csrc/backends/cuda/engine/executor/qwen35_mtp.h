#pragma once

#include "engine/models/qwen35/predictor.h"
#include "engine/state/qwen35_mtp.h"
#include "models/common/mtp.h"

#include <memory>

namespace mfq { class ModelSource; }
struct CudaExecutionContext;

namespace mfq::cuda::qwen35 {

class Mtp final : public MtpModule {
public:
    static std::unique_ptr<Mtp> load(const mfq::ModelSource& source,
        const mfq::models::qwen35::Config& config, CudaExecutionContext& execution);

    void reset(int64_t batch = 1) override;
    mfq_tensor_backend::Tensor forward(const MtpTarget& target,
        mfq_tensor_backend::Tensor hidden, mfq_tensor_backend::Tensor next_ids) override;
    MtpStep step_positioned(const MtpTarget& target, mfq_tensor_backend::Tensor hidden,
        mfq_tensor_backend::Tensor next_ids, mfq_tensor_backend::Tensor positions) override;

    int64_t cache_position() const noexcept override { return state_.position; }
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
    Mtp(Predictor model, CudaExecutionContext& execution);
    mfq_tensor_backend::Tensor evaluate(const MtpTarget& target,
        mfq_tensor_backend::Tensor hidden, mfq_tensor_backend::Tensor next_ids,
        mfq_tensor_backend::Tensor positions);
    mfq_tensor_backend::Tensor normalize(mfq_tensor_backend::Tensor input,
        const mfq_tensor_backend::Tensor& weight) const;

    Predictor model_;
    MtpState state_;
    std::vector<AttentionScratch> scratch_;
    CudaExecutionContext& execution_;
};

} // namespace mfq::cuda::qwen35
