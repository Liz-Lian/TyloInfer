#pragma once

#include "step.h"
#include "engine/generation/preparation.h"
#include "engine/config.h"
#include "engine/executor.h"
#include "engine/models/common/prompt.h"
#include "engine/models/metadata.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tyloi::engine {

inline bool batch_compatible(const EngineRequest &request, bool persistent_prefix, bool has_mtp) {
    const auto &input = request.input;
    return !input.media && input.cache_plan.session_id.empty() &&
           !((request.token_ids.empty() || input.cache_plan.stable_prefix_tokens) &&
               persistent_prefix) &&
           !(input.sampling.enable_mtp && has_mtp);
}

/** Prepare media within its resource owner, then advance the shared text generation flow. */
template <class Operations, class State, class Execution>
Generation generate_prepared(Operations operations, State &state, Execution& executor,
                             InferenceRequest &input, InferenceOutput &output,
                             const RequestId& id, bool batched) {
    using Ops = std::unwrap_reference_t<Operations>;
    Ops& ops = operations;
    std::optional<typename Ops::Prepared> prepared;
    if (input.vision && !output.stopped()) {
        auto job = ops.prepare(input, state);
        std::optional<Preparation> preparation;
        if constexpr (std::is_base_of_v<tyloi::Executor, std::remove_cvref_t<Execution>>)
            preparation.emplace(std::move(job), executor, ops.weights);
        while (!output.stopped()) {
            StepResult<typename Ops::Prepared> step;
            {
                auto timer = executor.timer();
                if constexpr (std::is_base_of_v<tyloi::Executor, std::remove_cvref_t<Execution>>)
                    step = preparation->next();
                else {
                    step = job.next();
                    if (step.workspace || step.graph || step.layer) throw std::logic_error("resource requests require a public executor");
                }
                timer->stop();
                output.metrics.multimodal_ms += timer->elapsed_ms();
            }
            if (step.value) {
                prepared = std::move(*step.value);
                break;
            }
            if (!step) throw std::runtime_error("media preparation returned no prompt");
            co_yield step.state;
        }
        if (output.stopped()) co_return;
        const auto elapsed = output.metrics.multimodal_ms;
        co_yield PrefillProgress{{0, 0.0, elapsed, elapsed}};
    }
    if (output.stopped()) co_return;
    auto sequence = ops.generate_text(input, output, state, std::move(prepared), id, batched);
    while (auto step = sequence.next()) {
        auto& event = step.value;
        if (auto *progress = event ? std::get_if<PrefillProgress>(&*event) : nullptr) {
            progress->timing.multimodal_ms = output.metrics.multimodal_ms;
            progress->timing.model_ms += progress->timing.multimodal_ms;
        }
        co_yield std::move(step);
    }
}

namespace detail {

// Session transactions surround both ordinary and speculative execution.
template <class Ops>
Generation generate_session(Ops& ops, InferenceRequest& request, InferenceOutput& output,
                            const GenerationPlan& plan) {
    using Hidden = typename Ops::Hidden;
    auto& state = ops.state;
    auto& session_cache = ops.cache;
    auto* mtp = ops.mtp;
    const auto& prepared = ops.prepared;
    const auto& prompt = request.prompt;
    const auto& sampling = request.sampling;
    const auto& cache_plan = request.cache_plan;
    const std::string input_key = prepared ? prepared->cache_key : std::string{};
    const bool caching =
        !ops.batched() && state.supports_text_session_state() &&
        (!prepared || !prepared->transformed() || !input_key.empty()) &&
        plan.stable_prefix_tokens > 0 &&
        (!cache_plan.session_id.empty() || session_cache.persistent_prefix_enabled());
    const bool use_mtp = mtp && sampling.enable_mtp && sampling.max_tokens > 1;
    const bool mtp_session = [&] {
        if (!use_mtp) return false;
#ifdef TYLOINFER_NATIVE_CUDA
        if constexpr (requires { state.predictor; })
            return state.predictor && mtp->supports_session_state(*state.predictor);
        else return false;
#else
        return mtp->supports_session_state();
#endif
    }();
    if constexpr(requires { ops.set_prefix_boundary(size_t{}); })
        ops.set_prefix_boundary(caching?std::min(plan.stable_prefix_tokens,prompt.size()-1):0);
    typename Ops::Restore restored;
    if (caching && (!use_mtp || mtp_session))
        restored = ops.restore_session(
            cache_plan.session_id,
            prompt,
            plan.stable_prefix_tokens,
            input_key,
            use_mtp);
    if (restored.tokens || use_mtp)
        ops.invalidate_plan();
    if (!restored.tokens) ops.reset();
    std::vector<int64_t> history = prompt;
    Hidden last_target_hidden;
    std::size_t last_snapshot = 0;
    bool initial_snapshot_attempted = false;
    const auto snapshot = [&] {
        const auto position = state.position();
        if (!caching || (use_mtp && !mtp_session) || output.result.cancelled || position <= 0 ||
            static_cast<std::size_t>(position) > history.size() ||
            static_cast<std::size_t>(position) == last_snapshot)
            return;
        try {
            if constexpr(requires { ops.prepare_snapshot(); })
                if(!ops.prepare_snapshot())return;
            std::vector<int64_t> tokens(history.begin(), history.begin() + position);
            if constexpr (requires { state.capture_text_session_state(tokens); }) {
                auto saved = state.capture_text_session_state(tokens);
                saved.input_key = input_key;
                if (use_mtp) {
                    if (!ops.has_hidden(last_target_hidden) || position <= 1)
                        return;
                    if constexpr (requires { state.predictor; })
                        saved.mtp = mtp->capture_session_state(*state.predictor, position, last_target_hidden);
                    else
                        throw std::invalid_argument("model state has no predictor state");
                    saved.bytes += saved.mtp->bytes;
                }
                session_cache.store(cache_plan.session_id, std::move(saved));
                last_snapshot = static_cast<std::size_t>(position);
            } else {
                throw std::runtime_error("text session state is unsupported by this model");
            }
        } catch (const std::exception &error) {
            std::cerr << "runtime_session_cache action=skip error=" << error.what() << '\n';
        }
    };
    auto sequence = use_mtp
        ? ops.speculate(request, output, restored.tokens, restored.mtp_last_target_hidden,
                        &last_target_hidden)
        : ops.plain(request, output, restored.tokens, caching ? plan.stable_prefix_tokens : 0);
    bool traced_decode_boundary = false;
    while (auto step = sequence.next()) {
        auto& event = step.value;
        if (auto *progress = event ? std::get_if<PrefillProgress>(&*event) : nullptr;
            progress && !use_mtp && caching &&
            progress->timing.prompt_tokens + restored.tokens == plan.stable_prefix_tokens)
            snapshot();
        if (auto *progress = event ? std::get_if<PrefillProgress>(&*event) : nullptr;
            progress && progress->session_boundary && use_mtp && caching &&
            !initial_snapshot_attempted) {
            initial_snapshot_attempted = true;
            snapshot();
        }
        if (auto *delta = event ? std::get_if<OutputDelta>(&*event) : nullptr) {
            if (!traced_decode_boundary && !delta->token_ids.empty()) {
                if constexpr (requires { ops.trace_expert_cache("decode_begin", false); })
                    ops.trace_expert_cache("decode_begin", false);
                traced_decode_boundary = true;
            }
            history.insert(history.end(), delta->token_ids.begin(), delta->token_ids.end());
            // Compatibility with speculative producers lacking an explicit
            // pre-token prompt boundary event.
            if (use_mtp && caching && !initial_snapshot_attempted) {
                initial_snapshot_attempted = true;
                snapshot();
            }
        }
        co_yield std::move(step);
    }
    // Flush parser-held text before a final session snapshot can copy or evict
    // a large cache. The scheduler's terminal finish remains idempotent and
    // still reports cleanup/cache completion after this final text event.
    if(caching) {
        auto tail=output.finish();
        if(!tail.diffs.empty())co_yield std::move(tail);
    }
    snapshot();
}

} // namespace detail

/** Validate and advance a request, retiring its operation owner after child borrows end. */
template <class Operations>
Generation generate_request(Operations operations, InferenceRequest& request, InferenceOutput& output) {
    std::unwrap_reference_t<Operations>& ops = operations;
    const auto& info = ops.info;
    const auto& config = ops.config;
    const auto& prepared = ops.prepared;
    const auto& prompt = request.prompt;
    auto& sampling = request.sampling;
    const auto& cache_plan = request.cache_plan;
    if (config.generation.prefill_chunk_size <= 0)
        throw std::invalid_argument("prefill chunk size must be positive");
    if (prepared && prepared->token_ids != prompt)
        throw std::invalid_argument("prepared prompt token IDs disagree");
    if constexpr (requires { ops.validate_prepared(); }) ops.validate_prepared();
    const auto occupied =
        static_cast<int64_t>(prompt.size()) +
        (prepared && prepared->transformed() ? prepared->decode_position_delta : 0);
    const auto plan = plan_generation(prompt,
        info.vocab_size,
        info.max_context,
        sampling.max_tokens,
        cache_plan.stable_prefix_tokens,
        std::max<int64_t>(prompt.size(), occupied));
    sampling.max_tokens = plan.generation_tokens;
    if (output.stopped())
        co_return;
    if constexpr (requires { ops.mtp_state; }) {
        if (ops.mtp) {
            ops.mtp_state.stats = {};
            ops.mtp_state.stats.available = true;
        }
    }
    ExecutionCleanup cleanup{output.cleanup_failure, ops};
    if constexpr (requires { ops.prepare_request(request); }) ops.prepare_request(request);
    {
        auto sequence = [&] {
            if constexpr (requires { ops.cache; })
                return detail::generate_session(ops, request, output, plan);
            else {
                ops.reset();
                return ops.plain(request, output, 0, 0);
            }
        }();
        while (auto step = sequence.next()) co_yield std::move(step);
    }
    cleanup.finish();
}

} // namespace tyloi::engine
