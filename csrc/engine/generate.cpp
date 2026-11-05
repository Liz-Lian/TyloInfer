#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/impl.h"
#include "engine/generation/step.h"
#include "engine/generation/sampling.h"
#include "engine/generation/prefill_budget.h"
#include "engine/scheduler/request.h"
#include "engine/models/common/mtp.h"
#include "engine/models/source.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/weights/routed_cache.h"
#include "engine/weights/moe/expert_cache.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace tyloi::engine {
using namespace tyloi::cuda;

StepSequence<EventData> Engine::generate(const RequestId&, Request& request) {
    auto& input = request.input;
    auto& output = *request.output;
    auto& state = *request.state;
    std::optional<models::Prepared> prepared;
    if (input.vision && !output.stopped()) {
        auto preparation = model_->prepare(input.prompt, *input.vision, state);
        while (!output.stopped()) {
            StepResult<models::Prepared> step;
            {
                auto timer = executor_->timer();
                step = preparation.next();
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
    auto& weights = *impl_->weights_;
    auto& graph = *impl_->graph_;
    auto& cache = *impl_->cache_;
    const auto& info = impl_->model_info_;
    const auto& config = impl_->runtime_;
    const auto& forward_options = impl_->forward_options_;
    const auto& prompt = input.prompt;
    const auto& cache_plan = input.cache_plan;
    if (config.generation.prefill_chunk_size <= 0)
        throw std::invalid_argument("prefill chunk size must be positive");
    if (prepared && prepared->token_ids != prompt)
        throw std::invalid_argument("prepared prompt token IDs disagree");
    if (prepared && prepared->image_mask && prepared->image_mask->defined())
        throw std::invalid_argument("model cannot apply an image participation mask");
    const auto occupied = static_cast<int64_t>(prompt.size()) +
        (prepared && prepared->transformed() ? prepared->decode_position_delta : 0);
    const auto plan = plan_generation(prompt, info.vocab_size, info.max_context,
        input.sampling.max_tokens, cache_plan.stable_prefix_tokens,
        std::max<int64_t>(prompt.size(), occupied));
    input.sampling.max_tokens = plan.generation_tokens;
    if (output.stopped()) co_return;

    const auto trace = [&](const char* boundary, bool wait = true) {
        static const bool enabled = [] {
            const char* value = env_lookup("TYLOINFER_TRACE_MOE_SUMMARY");
            return value && value[0] && value[0] != '0';
        }();
        if (!enabled || std::uncaught_exceptions()) return;
        std::cerr << "moe_request_cache_stats boundary=" << boundary << '\n';
        if (wait) print_moe_expert_cache_stats(weights.moe, std::cerr);
        std::cerr << "moe_phase_metrics boundary=" << boundary;
        for (const auto& [name, value] : weights.metrics())
            if (name == "ssd_expert_read_wait_ms" || name == "expert_route_wait_ms" ||
                name == "expert_host_fetch_ms" || name == "expert_host_copy_ms" ||
                name == "expert_transfer_cache_prepare_ms" || name == "ssd_expert_bytes_read" ||
                name == "ssd_expert_read_count" || name == "ram_expert_cache_hits" ||
                name == "ram_expert_cache_misses" || name == "ram_expert_pcie_read_bytes" ||
                name == "expert_logical_transfer_bytes") std::cerr << ' ' << name << '=' << value;
        std::cerr << '\n';
        if (impl_->source_) {
            const auto io = impl_->source_->file_read_stats();
            std::cerr << "runtime_model_io_stats boundary=" << boundary << " calls=" << io.calls
                      << " logical_bytes=" << io.logical_bytes << " physical_bytes=" << io.physical_bytes
                      << " read_ms=" << double(io.read_nanoseconds) / 1e6 << '\n';
        }
    };
    ExecutionCleanup cleanup{output.cleanup_failure, [&] {
        trace("end");
        try { state.reset(1); }
        catch (...) { graph.invalidate(); throw; }
        if (std::uncaught_exceptions()) graph.invalidate();
    }};
    trace("begin");
    const std::string input_key = prepared ? prepared->cache_key : std::string{};
    const bool caching = state.supports_text_session_state() &&
        (!prepared || !prepared->transformed() || !input_key.empty()) && plan.stable_prefix_tokens > 0 &&
        (!cache_plan.session_id.empty() || cache.persistent_prefix_enabled());
    cuda::internal::TextSessionRestore restored;
    if (caching) restored = cache.restore_best(state, [&] { drain(); },
        static_cast<MtpModule*>(nullptr), cache_plan.session_id, prompt, plan.stable_prefix_tokens, input_key);
    if (restored.tokens) graph.invalidate();
    else state.reset(1);
    std::size_t history_size = prompt.size();
    const auto snapshot = [&] {
        if (caching && !output.result.cancelled && state.position() > 0 &&
            static_cast<std::size_t>(state.position()) <= history_size)
            std::cerr << "runtime_session_cache action=skip error=text session state is unsupported by this model\n";
    };

    const auto tokens = prompt.size() - restored.tokens;
    const auto limit = config.generation.prefill_chunk_size;
    const bool short_streamed = state.position() > 0 && tokens <= 128 && moe_expert_cache_ssd_experts(weights.moe);
    const auto reserve_workspace = [&](std::size_t bytes) {
        reserve_moe_workspace(weights.moe, bytes);
        auto context = default_context(mfq_current_cuda_device());
        if (context->available_memory() >= bytes) return;
        std::size_t hot = 0;
        for (const auto& metric : cache.metrics())
            if (metric.first == "prefix_cache_hot_bytes") hot = std::size_t(metric.second);
        const auto shortage = bytes - context->available_memory();
        const auto released = cache.trim_hot(hot > shortage ? hot - shortage : 0);
        context->reclaim_idle_memory();
        std::cerr << "runtime_workspace_budget requested_bytes=" << bytes
                  << " available_bytes=" << context->available_memory()
                  << " prefix_released_bytes=" << released << '\n';
    };
    if (short_streamed) {
        std::cerr << "runtime_streamed_prefill retained_decode=1 tokens=" << tokens
                  << " chunk=" << std::min<int64_t>(4, limit) << '\n';
    } else {
        const bool streamed = tokens > 8 && moe_expert_cache_ssd_experts(weights.moe);
        if (streamed) lend_moe_decode_cache_for_prefill(weights.moe, runtime_options::workspace_reserve_bytes());
        if (tokens > 1024 || moe_expert_cache_ssd_experts(weights.moe))
            reserve_workspace(runtime_options::workspace_reserve_bytes());
    }
    int64_t chunk_size;
    if (short_streamed) chunk_size = std::min<int64_t>(4, limit);
    else if (tokens <= 128) chunk_size = std::min<int64_t>(std::max<std::size_t>(1, tokens), limit);
    else {
        auto budget = moe_prefill_workspace_budget(weights.moe, false);
        auto context = default_context(mfq_current_cuda_device());
        auto available = prefill_budget_add(prefill_budget_add(context->available_memory(), context->workspace_available()), budget.lendable_bytes);
        const auto reserve = moe_expert_cache_ssd_experts(weights.moe)
            ? std::max(std::size_t(1) << 30, runtime_options::workspace_reserve_bytes()) : std::size_t(1) << 30;
        const auto fixed = prefill_budget_add(reserve, budget.transfer_bytes);
        const auto wanted = std::min<int64_t>({int64_t(tokens), limit, max_prefill_compute_chunk});
        const auto required = prefill_budget_add(fixed, std::size_t(wanted) * info.prefill_row_bytes);
        if (required > available) {
            reserve_workspace(required > budget.lendable_bytes ? required - budget.lendable_bytes : 0);
            budget = moe_prefill_workspace_budget(weights.moe, false);
            available = prefill_budget_add(context->available_memory() + context->workspace_available(), budget.lendable_bytes);
        }
        chunk_size = select_prefill_chunk(tokens, limit, available, fixed, info.prefill_row_bytes);
        if (!chunk_size) throw InferenceInputError(InferenceInputErrorCode::ContextLength,
            "insufficient CUDA workspace for prefill computation and expert transfer buffers");
        std::cerr << "runtime_prefill_plan mode=" << (config.prefill_chunk_auto ? "auto" : "limited")
                  << " tokens=" << tokens << " limit=" << limit << " compute_chunk=" << chunk_size
                  << " available_bytes=" << available << " lendable_bytes=" << budget.lendable_bytes
                  << " transfer_bytes=" << budget.transfer_bytes << '\n';
    }

    auto full_ids = ops::reshape(*ops::tensor(std::span<const int64_t>(prompt), Device(kCUDA, forward_options.primary)), {1, -1});
    graph.ensure_storage(info.vocab_size);
    Sampler sampler(input.sampling, full_ids->device());
    const bool penalties = sampler.has_penalties();
    if (penalties) { ops::fill(*graph.counts, 0); ops::add_token_counts(*graph.counts, *full_ids); }
    const auto sample = [&](std::shared_ptr<Tensor> logits) {
        logits = ops::reshape(*ops::contiguous(*logits), {1, -1});
        if (penalties) sampler.apply_penalties(*logits, *graph.counts);
        auto next = sampler.sample(*logits);
        if (input.token_constraint && !input.token_constraint->allows(ops::item(*next))) {
            auto masked = ops::constrain(*logits, input.token_constraint);
            auto device_masked = ops::contiguous(*ops::copy(*masked, logits->device()));
            next = sampler.sample(*device_masked);
            if (!input.token_constraint->allows(ops::item(*next)))
                throw std::runtime_error("CUDA constrained sampler returned an invalid token");
        }
        if (input.token_constraint) input.token_constraint->accept(ops::item(*next));
        return next;
    };
    const auto project = [&](const Tensor& hidden, bool capped) {
        auto last = ops::contiguous(*ops::cast(*ops::select(hidden, 1, -1), kFloat16));
        return model_->logits(*last, capped);
    };
    std::shared_ptr<Tensor> pending;
    double elapsed = 0;
    auto offset = static_cast<int64_t>(restored.tokens);
    while (offset < static_cast<int64_t>(prompt.size()) && !output.stopped()) {
        const auto end = offset < static_cast<int64_t>(caching ? plan.stable_prefix_tokens : 0)
            ? std::min<int64_t>(prompt.size(), plan.stable_prefix_tokens) : int64_t(prompt.size());
        const auto chunk = next_prefill_chunk(end, offset, chunk_size);
        {
            PrefillActivity activity;
            auto timer = executor_->timer();
            models::Input model_input;
            model_input.ids = ops::contiguous(*ops::slice(*full_ids, 1, chunk.offset, chunk.offset + chunk.count));
            model_input.advance = true;
            if (prepared && prepared->transformed()) {
                if (prepared->embeddings) model_input.embeddings = ops::contiguous(*ops::slice(*prepared->embeddings, 1, chunk.offset, chunk.offset + chunk.count));
                if (prepared->positions) model_input.positions = ops::contiguous(*ops::slice(*prepared->positions, -1, chunk.offset, chunk.offset + chunk.count));
            }
            auto hidden = model_->forward(std::move(model_input), state, forward_options);
            if (prepared && prepared->transformed()) state.set_position_delta(prepared->decode_position_delta);
            timer->stop();
            if (!output.stopped() && chunk.offset + chunk.count == int64_t(prompt.size()))
                pending = sample(project(*hidden, true));
            const auto ms = timer->elapsed_ms();
            elapsed += ms;
            if (config.decode_graph.trace) std::cerr << "runtime_prefill_chunk offset=" << chunk.offset
                << " tokens=" << chunk.count << " layer_major=0 short_windows=0 compute_chunk=" << chunk_size
                << " elapsed_ms=" << ms << " timing=before_cache_restore\n";
        }
        offset += chunk.count;
        if (caching && static_cast<std::size_t>(offset) == plan.stable_prefix_tokens) snapshot();
        const auto multimodal_ms = output.metrics.multimodal_ms;
        co_yield PrefillProgress{{static_cast<std::size_t>(offset - restored.tokens), elapsed, multimodal_ms, elapsed + multimodal_ms}};
    }
    if (output.stopped()) {
        if (caching) {
            auto tail = output.finish();
            if (!tail.diffs.empty()) co_yield std::move(tail);
        }
        snapshot();
        cleanup.finish();
        co_return;
    }
    auto token = ops::item(*pending);
    if (weights.moe) restore_graph_expert_space();
    const bool graph_active = config.decode_graph.enabled &&
        impl_->load_.dsv4_cpu_offload_layers.empty() && impl_->load_.dense_cpu_layer_count == 0 &&
        impl_->graph_allowed_ &&
        (!prepared || !prepared->transformed()) && !input.token_constraint && info.decode_graph &&
        graph.graph && !weights.moe &&
        input.sampling.max_tokens >= config.decode_graph.minimum_generation_tokens &&
        input.sampling.max_tokens <= graph.generated_capacity;
    bool graph_prepared = false;
    bool traced_decode = false;
    while (!output.stopped()) {
        auto delta = output.append(std::vector<int64_t>{token});
        if (penalties && !graph_active) ops::add_token_counts(*graph.counts, *ops::contiguous(*pending));
        history_size += delta.token_ids.size();
        if (!traced_decode && !delta.token_ids.empty()) {
            trace("decode_begin", false);
            traced_decode = true;
        }
        co_yield std::move(delta);
        if (output.stopped()) break;
        if (!graph_active) {
            models::Input model_input;
            model_input.ids = ops::reshape(*pending, {1, 1});
            model_input.advance = true;
            const auto* profiling = env_lookup("TYLOINFER_PROFILE_DECODE");
            auto profile = profiling && profiling[0] == '1' ? executor_->profile("runtime.decode") : nullptr;
            auto hidden = model_->forward(std::move(model_input), state, forward_options);
            pending = sample(project(*hidden, !(sampler.greedy() && !penalties && !input.token_constraint)));
            if (profile) profile->finish();
            token = ops::item(*pending);
        } else {
            auto scope = graph.graph->enter();
            if (!graph_prepared) {
                if (penalties) ops::add_token_counts(*graph.counts, *ops::contiguous(*pending));
                graph.initialize_inputs(*pending, state.position(), sampler.random_host());
                const auto requested = state.position() + input.sampling.max_tokens;
                const auto planned = decode_bucket(requested, info.max_context);
                const auto parts = ops::attention_parts(planned, graph.device);
                const auto sample_static = [&] {
                    models::Input model_input;
                    model_input.ids = graph.static_input; model_input.positions = graph.static_pos;
                    model_input.cache_positions = graph.static_pos; model_input.lengths = graph.static_len;
                    model_input.kv_length = planned; model_input.attention_parts = parts;
                    auto hidden = model_->forward(std::move(model_input), state, forward_options);
                    auto logits = ops::reshape(*ops::contiguous(*project(*hidden, !(sampler.greedy() && !penalties))), {1, -1});
                    if (penalties) sampler.apply_penalties(*logits, *graph.counts);
                    return sampler.greedy() ? ops::sample_greedy(*logits) :
                        ops::sample_stochastic(*logits, *graph.random, sampler.params().temperature, sampler.params().top_k, sampler.params().top_p);
                };
                auto captured = state.storage();
                const bool hit = graph.matches(planned, sampler.params(), sampler.greedy(), captured);
                if (hit) ++graph.reuses;
                else {
                    graph.invalidate();
                    try {
                        auto recurrent = state.graph_state();
                        const auto saved_position = state.position();
                        graph.graph->prepare(recurrent);
                        if (!recurrent.empty() || info.double_warmup) {
                            const auto restore = [&] {
                                graph.graph->restore();
                                if (state.position() != saved_position)
                                    throw std::runtime_error("static decode warmup changed cache position");
                            };
                            for (int pass = 0; pass < 2; ++pass) {
                                try { sample_static(); restore(); }
                                catch (...) { restore(); throw; }
                            }
                        }
                        graph.graph->begin();
                        auto next = sample_static();
                        if (penalties) ops::add_token_counts(*graph.counts, *ops::contiguous(*next));
                        graph.capture_next(std::move(next));
                        captured = state.storage();
                        graph.graph->end(captured);
                        graph.set_key(planned, sampler.params(), sampler.greedy());
                        ++graph.captures;
                    } catch (...) { graph.invalidate(); throw; }
                    report_cuda_memory(impl_->load_.report_cuda_memory, "runtime_graph_capture");
                }
                if (config.decode_graph.trace) std::cerr << "runtime_cuda_graph action=" << (hit ? "reuse" : "capture")
                    << " requested_len=" << requested << " planned_len=" << planned << " captures=" << graph.captures
                    << " reuses=" << graph.reuses << std::endl;
                graph_prepared = true;
            }
            if (!sampler.greedy()) graph.upload_random(sampler.next_random());
            graph.graph->replay();
            token = ops::item(*graph.static_next);
            state.advance(1);
        }
    }
    if (caching) {
        auto tail = output.finish();
        if (!tail.diffs.empty()) co_yield std::move(tail);
    }
    snapshot();
    cleanup.finish();
}
} // namespace tyloi::engine

#else
#include "engine/models/deepseek_v4/forward.h"
#include "engine/models/gemma4/forward.h"
#include "engine/models/glm5_next/forward.h"
#include "engine/models/glm_dsa/forward.h"
#include "engine/models/minicpmo45/text/forward.h"
#include "engine/models/minicpmo45/forward.h"
#include "engine/models/nemotron_h/forward.h"
#include "engine/models/qwen4_exp/generation.h"
#include "engine/models/deepseek_v41/generation.h"

#include "engine/generate_impl.h"

namespace tyloi::cuda::internal {
#define TYLOINFER_INSTANTIATE_FLOW(MODEL, STATE)                                                                \
    template Generation generate(MODEL &, STATE &, tyloi::cuda::Executor*, tyloi::cuda::Collectives&, \
        tyloi::cuda::Executor::Mode, CudaProfiler&, tyloi::cuda::RoutedCache&, bool&, bool, bool,       \
        std::function<void()>,                                                                      \
        const tyloi::models::Info &, const tyloi::ModelSource*,                                          \
        DecodeGraphCache &,                                                                        \
        TextSessionCache &,                                                                        \
        const CudaRuntimeConfig &,                                                                 \
        tyloi::engine::InferenceRequest &,                                                           \
        tyloi::engine::InferenceOutput &,                                                            \
        tyloi::engine::mtp::State &,                                                                  \
        MtpModule *,                                                                               \
        std::optional<PreparedPrompt>,                                                         \
        tyloi::engine::Batch*, std::string,                    \
        std::function<void()>);
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::minicpmo45::Model, tyloi::cuda::minicpmo45::State)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::minicpmo45::text::Model, tyloi::cuda::minicpmo45::text::State)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::gemma4::Model, tyloi::cuda::gemma4::State)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::glm_dsa::Model, tyloi::cuda::glm_dsa::State)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::glm5_next::Model, tyloi::cuda::glm5_next::State)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::qwen4_exp::Model, tyloi::cuda::qwen4_exp::State)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::deepseek_v4::Model, tyloi::cuda::deepseek_v4::State)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::deepseek_v41::Model, tyloi::cuda::deepseek_v41::State)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::nemotron_h::Model, tyloi::cuda::nemotron_h::State)
#undef TYLOINFER_INSTANTIATE_FLOW
} // namespace tyloi::cuda::internal
#endif
