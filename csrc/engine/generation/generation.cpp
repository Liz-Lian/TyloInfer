#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/engine.h"
#include "engine/budget.h"
#include "engine/generation/decode.h"
#include "engine/generation/batch.h"
#include "engine/generation/generation.h"
#include "engine/generation/sampling.h"
#include "engine/generation/prefill_budget.h"
#include "engine/generation/prefill.h"
#include "engine/generation/mtp/generation.h"
#include "engine/models/causal_model.h"
#include "engine/models/common/mtp.h"
#include "engine/models/session/snapshot.h"
#include "engine/scheduler/request.h"
#include "engine/models/source.h"
#include "engine/state/session/sessions.h"
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
    Tensor host_ids;

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
        if (prepared && prepared->image_mask)
            model_input.image_mask = ops::contiguous(ops::slice(prepared->image_mask, 1, chunk.offset, chunk.offset + chunk.count));
        auto result = engine::prefill(model, state, executor, weights, budget, forward_options,
            std::move(model_input), chunk_size, output);
        if (prepared && prepared->transformed()) state.set_position_delta(prepared->decode_position_delta);
        timer->stop();
        if (!output.stopped() && chunk.offset + chunk.count == int64_t(input.prompt.size()))
            pending = sample_logits(sampler, project(result.hidden, true), graph.counts, input.token_constraint);
        const auto ms = result.elapsed_ms >= 0 ? result.elapsed_ms : timer->elapsed_ms();
        if (config.decode_graph.trace) std::cerr << "runtime_prefill_chunk offset=" << chunk.offset
            << " tokens=" << chunk.count << " layer_major=0 short_windows=0 compute_chunk=" << chunk_size
            << " elapsed_ms=" << ms << " timing=before_cache_restore\n";
        return ms;
    }
    int64_t first_token() {
        const auto token = ops::item(pending);
        weights.prepare(::tyloi::weights::Cache::Stage::decode, executor.device(), source);
        graph_active = config.decode_graph.enabled &&
            info.graph == models::GraphOwner::executor &&
            (!prepared || !prepared->transformed()) && !input.token_constraint && info.decode_graph &&
            graph.graph && input.sampling.max_tokens >= config.decode_graph.minimum_generation_tokens &&
            input.sampling.max_tokens <= graph.generated_capacity;
        return token;
    }
    void accept(int64_t token) {
        if (penalties && !graph_active) ops::add_token_counts(graph.counts, ops::contiguous(pending));
        if (((info.row_prefetch && config.prefetch.decode) || info.graph == models::GraphOwner::model) &&
            !(prepared && prepared->transformed()) && !output.stopped()) {
            if (!host_ids) host_ids = ops::empty({1, 1}, TensorOptions().device(kCPU).dtype(kInt64));
            ops::fill(host_ids, token);
            if (info.row_prefetch && config.prefetch.decode) model.prefetch(state, host_ids);
        }
    }
    int64_t advance() {
        reserve(1);
        if (info.graph == models::GraphOwner::model && (!prepared || !prepared->transformed())) {
            models::Input step;
            step.ids = ops::reshape(pending, {1, 1});
            step.host_ids = host_ids;
            models::DecodeOptions decode;
            decode.routed = weights.decode() == ::tyloi::weights::Cache::Decode::routed;
            const bool greedy = sampler.greedy() && !penalties && !input.token_constraint;
            decode.projection = greedy ? models::DecodeOptions::Projection::greedy : models::DecodeOptions::Projection::logits;
            if (auto decoded = model.decode(std::move(step), state, forward_options, decode)) {
                if (greedy) {
                    if (!decoded->ids) throw std::runtime_error("model decode returned no token IDs");
                    pending = std::move(decoded->ids);
                } else {
                    if (!decoded->logits) throw std::runtime_error("model decode returned no logits");
                    pending = sample_logits(sampler, ops::reshape(ops::contiguous(decoded->logits), {1, -1}),
                        graph.counts, input.token_constraint);
                }
                return ops::item(pending);
            }
        }
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
    using Hidden = Tensor;
    using Restore = ::tyloi::state::Sessions::Restore;
    models::CausalModel& model;
    models::State& state;
    tyloi::Executor& executor;
    Budget& budget;
    Decode& graph;
    weights::Cache& weights;
    ::tyloi::state::Sessions& cache;
    const models::Info& info;
    const Config& config;
    const models::ForwardOptions& forward_options;
    const ModelSource* source;
    tyloi::engine::mtp::State& mtp_state;
    std::optional<Prepared> prepared;
    models::Mtp* mtp = nullptr;

    bool batched() const { return false; }
    Restore restore_session(const std::string& session_id, const std::vector<std::int64_t>& prompt,
        std::size_t maximum_prefix, const std::string& input_key, bool use_mtp) {
        return cache.restore(state, executor, weights, budget, use_mtp ? mtp : nullptr,
            session_id, prompt, maximum_prefix, input_key);
    }
    bool has_hidden(const Hidden& value) const { return value && value.defined(); }
    void invalidate_plan() { graph.invalidate(); }
    std::size_t prefix_boundary = 0;
    void set_prefix_boundary(std::size_t tokens) { prefix_boundary = tokens; }
    std::int64_t prefill_chunk(const InferenceRequest& input, std::size_t reused, std::size_t stable) {
        const auto remaining = input.prompt.size() - reused;
        const bool resumed = state.position() > 0;
        const auto mode = weights.prefill_mode(remaining, resumed);
        const bool transformed = prepared && (prepared->transformed() || bool(prepared->image_mask));
        const auto limit = config.generation.prefill_chunk_size;
        budget.prepare_prefill(executor.device(), remaining, limit, mode);
        models::PrefillOptions policy{static_cast<std::int64_t>(input.prompt.size()), limit, reused, stable,
            transformed, weights.prefill_mode(remaining, resumed) == ::tyloi::weights::Cache::Prefill::retained,
            weights.decode() == ::tyloi::weights::Cache::Decode::routed,
            weights.transfer() == ::tyloi::weights::Cache::Transfer::coordinated,
            weights.split_prefill(remaining)};
        const auto plan = model.prefill_plan(state, policy);
        return budget.chunk(executor.device(), remaining, limit, info.prefill_row_bytes,
            resumed, config.prefill_chunk_auto, plan.workspace);
    }
    Generation speculate(InferenceRequest& input, InferenceOutput& output, std::size_t reused,
        const Tensor& restored, Hidden* committed) {
        const auto chunk = prefill_chunk(input, reused, prefix_boundary);
        return tyloi::engine::mtp::generate(model, state, executor, budget, weights, mtp_state,
            forward_options, input, output, chunk, prepared ? &*prepared : nullptr,
            reused, restored, committed, prefix_boundary, source, config.profile_mtp, config.prefetch);
    }
    StepSequence<Prepared> prepare(InferenceRequest& input, models::State& target) {
        return model.prepare(input.prompt, *input.vision, target);
    }
    void validate_prepared() const {}
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
    void reset() {
        if (mtp && (!state.predictor || !state.predictor->belongs_to(*mtp)))
            throw std::invalid_argument("predictor state is unavailable or belongs to another model");
        for (const auto& [device, growth] : state.reset_growth(1))
            if (growth) budget.reserve_kv(device, prefill_budget_add(growth, std::size_t(64) << 20));
        state.reset(1);
        if (mtp) {
            for (const auto& [device, growth] : mtp->reset_growth(*state.predictor, 1))
                if (growth) budget.reserve_kv(device, prefill_budget_add(growth, std::size_t(64) << 20));
            mtp->reset(*state.predictor, 1);
        }
    }
    void release() {
        trace_expert_cache("end");
        try { reset(); }
        catch (...) { graph.invalidate(); throw; }
        if (std::uncaught_exceptions()) graph.invalidate();
    }
    Generation plain(InferenceRequest& input, InferenceOutput& output, std::size_t reused, std::size_t stable) {
        const auto& prompt = input.prompt;
        const auto chunk_size = prefill_chunk(input, reused, stable);
        auto full_ids = ops::reshape(ops::tensor(std::span<const int64_t>(prompt),
            executor.device()), {1, -1});
        graph.ensure_storage(info.vocab_size);
        Sampler sampler(input.sampling, full_ids.device());
        const bool penalties = sampler.has_penalties();
        if (penalties) { ops::fill(graph.counts, 0); ops::add_token_counts(graph.counts, full_ids); }
        Plain plain{model, state, executor, budget, graph, weights, info, config,
                    forward_options, source, input, output, prepared, chunk_size,
                    std::move(full_ids), std::move(sampler), penalties};
        const auto remaining = prompt.size() - reused;
        const auto logical = model.prefill_plan(state, {static_cast<std::int64_t>(prompt.size()), chunk_size, reused, stable,
            prepared && (prepared->transformed() || bool(prepared->image_mask)),
            weights.prefill_mode(remaining, state.position() > 0) == ::tyloi::weights::Cache::Prefill::retained,
            weights.decode() == ::tyloi::weights::Cache::Decode::routed,
            weights.transfer() == ::tyloi::weights::Cache::Transfer::coordinated,
            weights.split_prefill(remaining)}).span;
        auto sequence = generate_sequence(std::ref(plain), output, prompt.size(), reused, stable, logical);
        while (auto step = sequence.next()) co_yield std::move(step);
    }
};

} // namespace

StepSequence<EventData> Engine::generate(const RequestId& id, Request& request) {
    if (request.batched) {
        graph_->invalidate();
        idle_state_->release_graphs();
        if (auto* predictor = model_->mtp()) {
            if (!idle_state_->predictor || !idle_state_->predictor->belongs_to(*predictor))
                throw std::invalid_argument("predictor state is unavailable or belongs to another model");
            for (const auto& [device, bytes] : predictor->reset_growth(*idle_state_->predictor, 1))
                if (bytes) budget_->reserve_kv(device, prefill_budget_add(bytes, std::size_t(64) << 20));
            predictor->reset(*idle_state_->predictor, 1);
        }
        return batch_->generate(id, request.input, *request.output);
    }
    return generate_prepared(
        Native{*model_, *request.state, *executor_, *budget_, *graph_, *weights_, *cache_, model_info_,
               config_, forward_options_, source_.get(), mtp_state_, {}, model_->mtp()},
        *request.state, *executor_, request.input, *request.output, id, request.batched);
}
} // namespace tyloi::engine
#endif
