#pragma once

#include "engine/models/qwen35/predictor.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/qwen35/mtp_state.h"
#include "engine/models/common/mtp.h"

#include <memory>

namespace tyloi::cuda::qwen35 {


class Mtp final : public tyloi::models::Mtp {
public:
    /** Own predictor weights and placement; execution options arrive with each step. */
    Mtp(Predictor model, ::tyloi::Tensor embedding, tyloi::ops::RopeTable rope,
        int device);

    std::unique_ptr<tyloi::models::MtpState> make_state() const override;
    void reset(tyloi::models::MtpState& state, int64_t batch = 1) const override;
    /** Advance the predictor using the request's current capacity and trace policy. */
    tyloi::models::MtpStep step(tyloi::models::MtpState& state, const ::tyloi::Tensor& hidden,
        const ::tyloi::Tensor& next_ids, const ::tyloi::Tensor& positions = {},
        const tyloi::models::ForwardOptions& options = {}) const override;
    /** Plan predictor KV backing for the actual step geometry and capacity. */
    std::vector<std::pair<tyloi::Device, std::size_t>> growth(
        const tyloi::models::MtpState& state, const tyloi::models::Mtp::Input& input) const override;
    /** Validate and retain a predictor session restore candidate. */
    std::unique_ptr<tyloi::models::Restore> prepare_restore(
        tyloi::models::MtpState& state, const MtpSessionState& snapshot) const override;

    void trim_cache_to(tyloi::models::MtpState& state, int64_t position) const override;
    bool supports_session_state() const noexcept override { return true; }
    MtpSessionState capture_session_state(const tyloi::models::MtpState& state, int64_t target_cache_position,
        const ::tyloi::Tensor& last_target_hidden) const override;

    bool teacher_forced_prompt_prime() const noexcept override { return true; }
    bool target_bootstrap_decode() const noexcept override { return false; }
    bool preserve_output_dtype() const noexcept override { return false; }
    bool retains_partial_target_prefix() const noexcept override { return true; }

private:

    ::tyloi::Tensor evaluate(MtpState& state, const ::tyloi::Tensor& hidden,
        const ::tyloi::Tensor& next_ids, const ::tyloi::Tensor& positions,
        const tyloi::models::ForwardOptions& options) const;

    Predictor model_;
    ::tyloi::Tensor embedding_;
    tyloi::ops::RopeTable rope_;
    int device_;
};

} // namespace tyloi::cuda::qwen35
