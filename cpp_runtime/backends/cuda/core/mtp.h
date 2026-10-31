#pragma once

#include "storage/session_state.h"
#include "mtp_policy.h"
#include "mfq_tensor_backend.h"
#include "core/causal_model.h"

#include <cstdint>
#include <variant>
#include <stdexcept>
#include <utility>
#include <vector>
#include <functional>
#include <optional>

struct RopeCache;

struct MtpTarget {
    std::variant<mfq::cuda::Qwen35CausalLm*, mfq::cuda::MiniCPMO45CausalLm*,
        mfq::cuda::MiniCPMOTtsCausalLm*, mfq::cuda::Gemma4CausalLm*,
        mfq::cuda::GlmDsaCausalLm*, mfq::cuda::Glm5CausalLm*, mfq::cuda::Qwen4CausalLm*,
        mfq::cuda::DeepseekV4CausalLm*, mfq::cuda::DeepseekV41CausalLm*,
        mfq::cuda::NemotronHCausalLm*> model;
    const RopeCache* rope = nullptr;
    template <class Model> explicit MtpTarget(Model& target) : model(&target), rope(&target.rope) {}
    mfq_tensor_backend::Tensor embed(mfq_tensor_backend::Tensor ids) const;
    mfq_tensor_backend::Tensor logits(mfq_tensor_backend::Tensor hidden) const;
};

struct MtpStep {
    mfq_tensor_backend::Tensor sample_hidden;
    mfq_tensor_backend::Tensor chain_hidden;
    // Optional last-row vocabulary projection computed in the predictor graph.
    mfq_tensor_backend::Tensor sample_logits;
    int32_t greedy_token=-1;
    float greedy_probability=-1.f;
};

struct MtpGreedyChain {
    std::vector<int32_t> tokens;
    std::vector<float> probabilities;
};

// Architecture-specific predictors implement device math and cache state only.
struct MtpModule {
    virtual ~MtpModule() = default;
    virtual void reset(int64_t batch = 1) = 0;
    virtual mfq_tensor_backend::Tensor forward(
        const MtpTarget& target,
        mfq_tensor_backend::Tensor previous_hidden,
        mfq_tensor_backend::Tensor next_ids) = 0;
    virtual MtpStep step(
        const MtpTarget& target,
        mfq_tensor_backend::Tensor previous_hidden,
        mfq_tensor_backend::Tensor next_ids) {
        auto hidden = forward(
            target, std::move(previous_hidden), std::move(next_ids));
        return {hidden, hidden};
    }
    virtual MtpStep step_positioned(
        const MtpTarget&,
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
    // Additional transient work beyond the target's sampling/history budget.
    // KV, weights and captured buffers are already charged. An unprepared or
    // unpriced predictor keeps the engine's conservative capture allowance.
    virtual std::optional<std::size_t> extra_decode_workspace_bytes(int) const { return {}; }
    std::function<void()> decode_workspace_changed;
    // Preferred committed-context batch during ordinary decoding. The engine
    // flushes pending rows before drafting or publishing a final checkpoint.
    virtual int preferred_context_update_batch_size() const noexcept { return 1; }
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
    // Decode commits may combine projection with the next draft. Prompt
    // priming uses append_target_context so its work stays in prefill timing.
    virtual void append_target_context_for_draft(mfq_tensor_backend::Tensor hidden,std::int64_t start) {
        append_target_context(std::move(hidden),start);
    }
    virtual mfq_tensor_backend::Tensor draft_block(
        const MtpTarget&,
        mfq_tensor_backend::Tensor,
        int) {
        throw std::runtime_error(
            "this CUDA MTP predictor has no block-draft adapter");
    }
    // An optional greedy walk over the block's compact candidate rows.
    // Empty keeps the generic per-row sampler (including its penalties).
    virtual std::vector<int32_t> draft_greedy_tokens(mfq_tensor_backend::Tensor) { return {}; }
    virtual MtpGreedyChain draft_greedy_chain(mfq_tensor_backend::Tensor anchor,bool probabilities) {
        if(probabilities)return {};
        return {draft_greedy_tokens(std::move(anchor)),{}};
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
