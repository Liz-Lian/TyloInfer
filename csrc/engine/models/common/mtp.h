#pragma once

#include "engine/models/session.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/generation/mtp/policy.h"
#include "engine/requests/sampling.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

struct MtpStep {
    ::tyloi::Tensor sample_hidden;
    ::tyloi::Tensor chain_hidden;
    // Optional last-row vocabulary projection computed in the predictor graph.
    ::tyloi::Tensor sample_logits;
    int32_t greedy_token=-1;
};

// Predictors provide model math, cache state and speculative policy behavior.
struct MtpModule {
    virtual ~MtpModule() = default;
    virtual void reset(int64_t batch = 1) = 0;
    virtual ::tyloi::Tensor forward(
        ::tyloi::Tensor previous_hidden,
        ::tyloi::Tensor next_ids) = 0;
    virtual MtpStep step(
        ::tyloi::Tensor previous_hidden,
        ::tyloi::Tensor next_ids) {
        auto hidden = forward(
            std::move(previous_hidden), std::move(next_ids));
        return {hidden, hidden};
    }
    virtual MtpStep step_positioned(
        ::tyloi::Tensor,
        ::tyloi::Tensor,
        ::tyloi::Tensor) {
        throw std::runtime_error(
            "this CUDA MTP predictor does not accept explicit positions");
    }
    virtual int64_t cache_position() const noexcept = 0;
    virtual void trim_cache_to(int64_t position) = 0;
    virtual bool teacher_forced_prompt_prime() const noexcept = 0;
    virtual bool target_bootstrap_decode() const noexcept = 0;
    virtual bool preserve_output_dtype() const noexcept = 0;
    virtual bool take_graph_initialization() noexcept { return false; }
    virtual uint64_t graph_replay_count() const noexcept { return 0; }
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
    virtual MtpSessionState capture_session_state(
        std::int64_t,
        const ::tyloi::Tensor&) const {
        throw std::runtime_error(
            "this CUDA MTP predictor has no session state");
    }
    virtual void restore_session_state(const MtpSessionState&) {
        throw std::runtime_error(
            "this CUDA MTP predictor has no session state");
    }
    virtual void seed_target_context(::tyloi::Tensor hidden, std::int64_t tokens) {
        TYLOINFER_RUNTIME_CHECK(hidden.dim() == 3 && hidden.size(1) == tokens,
                          "this CUDA MTP predictor requires complete prompt context");
        append_target_context(std::move(hidden), 0);
    }
    virtual void append_target_context(
        ::tyloi::Tensor,
        std::int64_t) {
        throw std::runtime_error(
            "this CUDA MTP predictor has no target-context adapter");
    }
    virtual ::tyloi::Tensor draft_block(
        ::tyloi::Tensor,
        int) {
        throw std::runtime_error(
            "this CUDA MTP predictor has no block-draft adapter");
    }
    virtual ::tyloi::Tensor draft_next(
        ::tyloi::Tensor,
        ::tyloi::Tensor) {
        throw std::runtime_error("this CUDA MTP predictor has no block-draft adapter");
    }

};
