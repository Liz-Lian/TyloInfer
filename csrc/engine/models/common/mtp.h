#pragma once

#include "engine/models/session/snapshot.h"
#include "engine/tensor.h"
#include "engine/models/state.h"
#include "engine/generation/mtp/policy.h"
#include "engine/requests/sampling.h"

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace tyloi::models {

struct MtpStep {
    Tensor sample_hidden;
    Tensor chain_hidden;
    // Optional last-row vocabulary projection computed in the predictor graph.
    Tensor sample_logits;
    int32_t greedy_token=-1;
};

// Predictor weights and math use the explicit state owned by the request.
struct Mtp {
    virtual ~Mtp() = default;
    virtual std::unique_ptr<MtpState> make_state() const = 0;
    virtual void reset(MtpState& state, int64_t batch = 1) const = 0;
    /** Return per-device backing that resetting predictor state will newly allocate. */
    virtual std::vector<std::pair<Device, std::size_t>> reset_growth(const MtpState&, int64_t) const { return {}; }
    /** Estimate device backing allocated by the next predictor step. */
    virtual std::vector<std::pair<Device, std::size_t>> growth(
        const MtpState&, const Tensor&) const {
        throw std::logic_error("predictor allocation planning is unsupported");
    }
    virtual MtpStep step(
        MtpState& state,
        const Tensor& previous_hidden,
        const Tensor& next_ids,
        const Tensor& positions = {}) const = 0;
    virtual void trim_cache_to(MtpState& state, int64_t position) const = 0;
    virtual bool teacher_forced_prompt_prime() const noexcept = 0;
    virtual bool target_bootstrap_decode() const noexcept = 0;
    virtual bool preserve_output_dtype() const noexcept = 0;
    virtual int maximum_draft_depth() const noexcept {
        return tyloi::engine::mtp::kMaximumDraftDepth;
    }
    virtual bool blockwise_drafting() const noexcept { return false; }
    virtual std::unique_ptr<tyloi::engine::mtp::DepthPolicy> depth_policy(
            int maximum_depth, tyloi::engine::mtp::PolicyState& state) const {
        return std::make_unique<tyloi::engine::mtp::AdaptivePolicy>(maximum_depth, &state);
    }
    virtual MfqSamplingParams draft_sampling(MfqSamplingParams sampling, bool) const {
        return sampling;
    }
    virtual bool split_target_verification() const noexcept { return false; }
    virtual bool retains_partial_target_prefix() const noexcept {
        return false;
    }
    virtual bool supports_session_state() const noexcept { return false; }
    /** Validate the predictor's committed position against its target's session boundary. */
    virtual bool session_boundary(std::int64_t target, std::int64_t predictor) const noexcept {
        return target > 1 && predictor == target - 1;
    }
    virtual MtpSessionState capture_session_state(
        const MtpState&,
        std::int64_t,
        const Tensor&) const {
        throw std::runtime_error(
            "this CUDA MTP predictor has no session state");
    }
    virtual void restore_session_state(MtpState&, const MtpSessionState&) const {
        throw std::runtime_error(
            "this CUDA MTP predictor has no session state");
    }
    /** Validate and own a staged restore, or return null when joint restoration is unavailable. */
    virtual std::unique_ptr<Restore> prepare_restore(MtpState&, const MtpSessionState&) const { return {}; }
    virtual void seed_target_context(MtpState& state, const Tensor& hidden, std::int64_t tokens) const {
        if (hidden.dim() != 3 || hidden.size(1) != tokens)
            throw std::runtime_error("this CUDA MTP predictor requires complete prompt context");
        append_target_context(state, hidden, 0);
    }
    virtual void append_target_context(
        MtpState&,
        const Tensor&,
        std::int64_t) const {
        throw std::runtime_error(
            "this CUDA MTP predictor has no target-context adapter");
    }
    virtual Tensor draft_block(
        MtpState&,
        const Tensor&,
        int) const {
        throw std::runtime_error(
            "this CUDA MTP predictor has no block-draft adapter");
    }
    virtual Tensor draft_next(
        const Tensor&,
        const Tensor&) const {
        throw std::runtime_error("this CUDA MTP predictor has no block-draft adapter");
    }

protected:
    template<class T> T& state(MtpState& value) const {
        if (!value.belongs_to(*this)) throw std::invalid_argument("predictor state belongs to another model");
        return static_cast<T&>(value);
    }
    template<class T> const T& state(const MtpState& value) const {
        if (!value.belongs_to(*this)) throw std::invalid_argument("predictor state belongs to another model");
        return static_cast<const T&>(value);
    }
};

}  // namespace tyloi::models
