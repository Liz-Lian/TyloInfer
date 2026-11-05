#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/impl.h"
#include "engine/generation/step.h"
#include "engine/generation/sampling.h"
#include "engine/scheduler/request.h"
#include "engine/models/source.h"
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
        if (!config.trace_moe || std::uncaught_exceptions()) return;
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
    state.reset(1);

    const auto tokens = prompt.size();
    const auto limit = config.generation.prefill_chunk_size;
    const auto chunk_size = impl_->budget_->prefill(tokens, limit, info.prefill_row_bytes,
        state.position() > 0, config.prefill_chunk_auto);

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
    int64_t offset = 0;
    while (offset < static_cast<int64_t>(prompt.size()) && !output.stopped()) {
        const auto chunk = next_prefill_chunk(int64_t(prompt.size()), offset, chunk_size);
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
        const auto multimodal_ms = output.metrics.multimodal_ms;
        co_yield PrefillProgress{{static_cast<std::size_t>(offset), elapsed, multimodal_ms, elapsed + multimodal_ms}};
    }
    if (output.stopped()) {
        cleanup.finish();
        co_return;
    }
    auto token = ops::item(*pending);
    if (weights.moe) impl_->budget_->decode(impl_->expert_space_ ? &*impl_->expert_space_ : nullptr, impl_->source_.get());
    const bool graph_active = config.decode_graph.enabled &&
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
            auto profile = config.profile_decode ? executor_->profile("runtime.decode") : nullptr;
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
