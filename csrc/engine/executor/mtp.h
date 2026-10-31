#pragma once

#include "engine/models/common/session/state.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/executor/mtp_policy.h"

#include <cstdint>
#include <stdexcept>
#include <utility>

struct MtpStep {
    mfq_tensor_backend::Tensor sample_hidden;
    mfq_tensor_backend::Tensor chain_hidden;
    // Optional last-row vocabulary projection computed in the predictor graph.
    mfq_tensor_backend::Tensor sample_logits;
    int32_t greedy_token=-1;
};

// Architecture-specific predictors implement device math and cache state only.
struct MtpModule {
    virtual ~MtpModule() = default;
    virtual void reset(int64_t batch = 1) = 0;
    virtual mfq_tensor_backend::Tensor forward(
        mfq_tensor_backend::Tensor previous_hidden,
        mfq_tensor_backend::Tensor next_ids) = 0;
    virtual MtpStep step(
        mfq_tensor_backend::Tensor previous_hidden,
        mfq_tensor_backend::Tensor next_ids) {
        auto hidden = forward(
            std::move(previous_hidden), std::move(next_ids));
        return {hidden, hidden};
    }
    virtual MtpStep step_positioned(
        mfq_tensor_backend::Tensor,
        mfq_tensor_backend::Tensor,
        mfq_tensor_backend::Tensor) {
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
        return mfq::engine::mtp::kMaximumDraftDepth;
    }
    virtual bool blockwise_drafting() const noexcept { return false; }
    virtual bool dspark_policy() const noexcept { return false; }
    virtual bool split_target_verification() const noexcept { return false; }
    virtual bool retains_partial_target_prefix() const noexcept {
        return false;
    }
    virtual bool supports_session_state() const noexcept { return false; }
    virtual MtpSessionState capture_session_state(
        std::int64_t,
        const mfq_tensor_backend::Tensor&) const {
        throw std::runtime_error(
            "this CUDA MTP predictor has no session state");
    }
    virtual void restore_session_state(const MtpSessionState&) {
        throw std::runtime_error(
            "this CUDA MTP predictor has no session state");
    }
    virtual void seed_target_context(mfq_tensor_backend::Tensor hidden, std::int64_t tokens) {
        TYLOINFER_RUNTIME_CHECK(hidden.dim() == 3 && hidden.size(1) == tokens,
                          "this CUDA MTP predictor requires complete prompt context");
        append_target_context(std::move(hidden), 0);
    }
    virtual void append_target_context(
        mfq_tensor_backend::Tensor,
        std::int64_t) {
        throw std::runtime_error(
            "this CUDA MTP predictor has no target-context adapter");
    }
    virtual mfq_tensor_backend::Tensor draft_block(
        mfq_tensor_backend::Tensor,
        int) {
        throw std::runtime_error(
            "this CUDA MTP predictor has no block-draft adapter");
    }
    virtual mfq_tensor_backend::Tensor draft_next(
        mfq_tensor_backend::Tensor,
        mfq_tensor_backend::Tensor) {
        throw std::runtime_error("this CUDA MTP predictor has no block-draft adapter");
    }

    mfq::engine::mtp::GenerationStats last_stats;
    mfq::engine::mtp::PolicyState policy_state;
    uint64_t last_cycles = 0;
    uint64_t last_accepted = 0;
    uint64_t last_rejected = 0;
};
