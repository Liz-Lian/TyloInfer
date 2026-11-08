#include "engine/weights/cache.h"
#include "engine/models/qwen35/forward.h"
#include "engine/models/qwen35/mtp.h"

#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/engine.h"
#include "engine/budget.h"
#include "engine/generation/decode.h"
#include "engine/generation/generation.h"
#include "engine/generation/sampling.h"
#include "engine/generation/prefill_budget.h"
#include "engine/models/causal_model.h"
#include "engine/scheduler/request.h"
#include "engine/models/source.h"
#include "engine/weights/cache.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace tyloi::engine {
namespace {

struct Plain {
    models::CausalModel& model;
    models::State& state;
    tyloi::Executor& executor;
    Budget& budget;
    Decode& graph;
    weights::Cache& weights;
    const models::Info& info;
    const Config& config;
    const models::ForwardOptions& forward_options;
    const ModelSource* source;
    InferenceRequest& input;
    InferenceOutput& output;
    const std::optional<models::Prepared>& prepared;
    int64_t chunk_size;
    Tensor full_ids;
    Sampler sampler;
    bool penalties;
    Tensor pending;
    bool graph_active = false;
    bool graph_prepared = false;

    Tensor project(const Tensor& hidden, bool capped) {
        auto last = ops::contiguous(ops::cast(ops::select(hidden, 1, -1), kFloat16));
        return model.logits(last, capped);
    }
    void reserve(int64_t tokens) {
        for (const auto& [device, growth] : state.growth(tokens, forward_options))
            if (growth) budget.reserve_kv(device, prefill_budget_add(growth, std::size_t(64) << 20));
    }
    double prefill(const PrefillChunk& chunk) {
        auto timer = executor.timer();
        models::Input model_input;
        model_input.ids = ops::contiguous(ops::slice(full_ids, 1, chunk.offset, chunk.offset + chunk.count));
        model_input.advance = true;
        if (prepared && prepared->transformed()) {
            if (prepared->embeddings) model_input.embeddings = ops::contiguous(ops::slice(prepared->embeddings, 1, chunk.offset, chunk.offset + chunk.count));
            if (prepared->positions) model_input.positions = ops::contiguous(ops::slice(prepared->positions, -1, chunk.offset, chunk.offset + chunk.count));
        }
        reserve(chunk.count);
        auto hidden = model.forward(std::move(model_input), state, forward_options);
        if (prepared && prepared->transformed()) state.set_position_delta(prepared->decode_position_delta);
        timer->stop();
        if (!output.stopped() && chunk.offset + chunk.count == int64_t(input.prompt.size()))
            pending = sample_logits(sampler, project(hidden, true), graph.counts, input.token_constraint);
        const auto ms = timer->elapsed_ms();
        if (config.decode_graph.trace) std::cerr << "runtime_prefill_chunk offset=" << chunk.offset
            << " tokens=" << chunk.count << " layer_major=0 short_windows=0 compute_chunk=" << chunk_size
            << " elapsed_ms=" << ms << " timing=before_cache_restore\n";
        return ms;
    }
    int64_t first_token() {
        const auto token = ops::item(pending);
        weights.prepare(::tyloi::weights::Cache::Stage::decode, Device(kCUDA, forward_options.primary), source);
        graph_active = config.decode_graph.enabled &&
            (!prepared || !prepared->transformed()) && !input.token_constraint && info.decode_graph &&
            graph.graph && input.sampling.max_tokens >= config.decode_graph.minimum_generation_tokens &&
            input.sampling.max_tokens <= graph.generated_capacity;
        return token;
    }
    void accept(int64_t) {
        if (penalties && !graph_active) ops::add_token_counts(graph.counts, ops::contiguous(pending));
    }
    int64_t advance() {
        reserve(1);
        if (!graph_active) {
            models::Input model_input;
            model_input.ids = ops::reshape(pending, {1, 1});
            model_input.advance = true;
            auto profile = config.profile_decode ? executor.profile("runtime.decode") : nullptr;
            auto hidden = model.forward(std::move(model_input), state, forward_options);
            pending = sample_logits(sampler,
                project(hidden, !(sampler.greedy() && !penalties && !input.token_constraint)),
                graph.counts, input.token_constraint);
            if (profile) profile->finish();
            return ops::item(pending);
        }
        auto scope = graph.graph->enter();
        if (!graph_prepared) {
            if (penalties) ops::add_token_counts(graph.counts, ops::contiguous(pending));
            graph.initialize_inputs(pending, state.position(), sampler.random_host());
            const auto requested = state.position() + input.sampling.max_tokens;
            const auto planned = decode_bucket(requested, info.max_context);
            const auto parts = ops::attention_parts(planned, graph.device);
            const auto sample_static = [&] {
                models::Input model_input;
                model_input.ids = graph.static_input; model_input.positions = graph.static_pos;
                model_input.cache_positions = graph.static_pos; model_input.lengths = graph.static_len;
                model_input.kv_length = planned; model_input.attention_parts = parts;
                auto hidden = model.forward(std::move(model_input), state, forward_options);
                auto logits = ops::reshape(ops::contiguous(project(hidden, !(sampler.greedy() && !penalties))), {1, -1});
                if (penalties) sampler.apply_penalties(logits, graph.counts);
                return sampler.greedy() ? ops::sample_greedy(logits) :
                    ops::sample_stochastic(logits, graph.random, sampler.params().temperature, sampler.params().top_k, sampler.params().top_p);
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
                    if (penalties) ops::add_token_counts(graph.counts, ops::contiguous(next));
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
        const auto token = ops::item(graph.static_next);
        state.advance(1);
        return token;
    }
};

struct Native {
    using Prepared = models::Prepared;
    models::CausalModel& model;
    models::State& state;
    tyloi::Executor& executor;
    Budget& budget;
    Decode& graph;
    weights::Cache& weights;
    const models::Info& info;
    const Config& config;
    const models::ForwardOptions& forward_options;
    const ModelSource* source;
    std::optional<Prepared> prepared;

    StepSequence<Prepared> prepare(InferenceRequest& input, models::State& target) {
        return model.prepare(input.prompt, *input.vision, target);
    }
    auto advance_preparation(StepSequence<Prepared>& preparation) {
        auto timer = executor.timer();
        auto step = preparation.next();
        timer->stop();
        return std::pair{std::move(step), timer->elapsed_ms()};
    }
    void validate_prepared() const {
        if (prepared && prepared->image_mask && prepared->image_mask->defined())
            throw std::invalid_argument("model cannot apply an image participation mask");
    }
    Generation generate_text(InferenceRequest& input, InferenceOutput& output, models::State&,
                             std::optional<Prepared> value, const RequestId&, bool) {
        prepared = std::move(value);
        return generate_request(std::ref(*this), input, output);
    }
    void trace_expert_cache(const char* boundary, bool wait = true) {
        if (!config.trace_moe || std::uncaught_exceptions()) return;
        weights.trace(boundary, wait, source);
    }
    void prepare_request(const InferenceRequest&) { trace_expert_cache("begin"); }
    void reset() { state.reset(1); }
    void release() {
        trace_expert_cache("end");
        try { reset(); }
        catch (...) { graph.invalidate(); throw; }
        if (std::uncaught_exceptions()) graph.invalidate();
    }
    Generation plain(InferenceRequest& input, InferenceOutput& output, std::size_t, std::size_t) {
        const auto& prompt = input.prompt;
        const auto chunk_size = budget.prefill(prompt.size(), config.generation.prefill_chunk_size,
            info.prefill_row_bytes, state.position() > 0, config.prefill_chunk_auto);
        auto full_ids = ops::reshape(*ops::tensor(std::span<const int64_t>(prompt),
            Device(kCUDA, forward_options.primary)), {1, -1});
        graph.ensure_storage(info.vocab_size);
        Sampler sampler(input.sampling, full_ids->device());
        const bool penalties = sampler.has_penalties();
        if (penalties) { ops::fill(graph.counts, 0); ops::add_token_counts(graph.counts, full_ids); }
        Plain plain{model, state, executor, budget, graph, weights, info, config,
                    forward_options, source, input, output, prepared, chunk_size,
                    std::move(full_ids), std::move(sampler), penalties};
        auto sequence = generate_sequence(std::ref(plain), output, prompt.size(), 0, 0, chunk_size);
        bool traced_decode = false;
        while (auto step = sequence.next()) {
            if (auto* delta = step.value ? std::get_if<OutputDelta>(&*step.value) : nullptr;
                delta && !traced_decode && !delta->token_ids.empty()) {
                trace_expert_cache("decode_begin", false);
                traced_decode = true;
            }
            co_yield std::move(step);
        }
    }
};

} // namespace

StepSequence<EventData> Engine::generate(const RequestId& id, Request& request) {
    return generate_prepared(
        Native{*model_, *request.state, *executor_, *budget_, *graph_, *weights_, model_info_,
               config_, forward_options_, source_.get()},
        *request.state, request.input, *request.output, id, request.batched);
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

#endif

#include "engine/generation/cuda_impl.h"

namespace tyloi::cuda::internal {
#define TYLOINFER_INSTANTIATE_FLOW(MODEL, STATE, PREDICTOR)                                                                \
    template Generation generate(MODEL &, STATE &, tyloi::cuda::Executor*, tyloi::cuda::Collectives&, \
        tyloi::cuda::Executor::Mode, CudaProfiler&, tyloi::weights::Cache&, tyloi::engine::Budget&, bool&, bool, bool, \
        std::function<void()>,                                                                      \
        const tyloi::models::Info &, const tyloi::ModelSource*,                                          \
        DecodeGraphCache &,                                                                        \
        tyloi::engine::SessionCache<tyloi::cuda::internal::CudaSessionOps> &,                       \
        const tyloi::engine::Config &,                                                      \
        tyloi::engine::InferenceRequest &,                                                           \
        tyloi::engine::InferenceOutput &,                                                            \
        tyloi::engine::mtp::State &,                                                                  \
        PREDICTOR *,                                                                      \
        std::optional<tyloi::models::Prepared>,                                                         \
        tyloi::engine::Batch*, std::string,                    \
        bool);
#ifndef TYLOINFER_NATIVE_CUDA
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::minicpmo45::Model, tyloi::cuda::minicpmo45::State, tyloi::models::Mtp)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::minicpmo45::text::Model, tyloi::cuda::minicpmo45::text::State, tyloi::models::Mtp)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::gemma4::Model, tyloi::cuda::gemma4::State, tyloi::models::Mtp)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::glm_dsa::Model, tyloi::cuda::glm_dsa::State, tyloi::models::Mtp)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::glm5_next::Model, tyloi::cuda::glm5_next::State, tyloi::models::Mtp)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::qwen4_exp::Model, tyloi::cuda::qwen4_exp::State, tyloi::models::Mtp)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::deepseek_v4::Model, tyloi::cuda::deepseek_v4::State, tyloi::models::Mtp)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::deepseek_v41::Model, tyloi::cuda::deepseek_v41::State, tyloi::models::Mtp)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::nemotron_h::Model, tyloi::cuda::nemotron_h::State, tyloi::models::Mtp)
#endif
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::qwen35::Model, tyloi::cuda::qwen35::State, tyloi::models::Mtp)
TYLOINFER_INSTANTIATE_FLOW(tyloi::cuda::qwen35::Model, tyloi::cuda::qwen35::State, tyloi::cuda::qwen35::Mtp)
#undef TYLOINFER_INSTANTIATE_FLOW
} // namespace tyloi::cuda::internal
