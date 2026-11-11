#include "engine/generation/decode.h"

#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/models/causal_model.h"
#include "engine/weights/cache.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>

namespace tyloi::engine {
Capture::Capture(Executor& executor, weights::Cache& weights) : executor_(executor), weights_(weights) {}
Capture::~Capture() = default;

void Capture::step(GraphPhase phase) {
    switch (phase) {
    case GraphPhase::prepare: weights_.retire_prefill(); break;
    case GraphPhase::ready: weights_.finish_exchange(); break;
    case GraphPhase::settle: executor_.synchronize(); break;
    case GraphPhase::warmup:
        if (warmup_) throw std::logic_error("duplicate graph warmup phase");
        warmup_ = std::make_unique<weights::Cache::Warmup>(weights_, true);
        break;
    case GraphPhase::target:
        if (!warmup_ || target_) throw std::logic_error("invalid graph target phase");
        warmup_->release(executor_.device(), true);
        target_ = true;
        break;
    default: throw std::logic_error("unknown graph phase");
    }
}
void Capture::finish() {
    if (warmup_) {
        if (!target_) throw std::logic_error("graph warmup ended before target phase");
        warmup_->release(executor_.device(), false);
        warmup_.reset();
    }
}
void Capture::drain() {
    std::exception_ptr failure;
    try { weights_.drain(); } catch (...) { failure=std::current_exception(); }
    try { executor_.drain(); } catch (...) { if (!failure) failure=std::current_exception(); }
    if (failure) std::rethrow_exception(failure);
}

namespace {
std::optional<models::Decoded> drive(StepSequence<models::Decoded> job, Executor& executor,
                                      weights::Cache& weights, models::State& state, bool result_required) {
    Capture capture(executor, weights);
    bool observed = false;
    try {
        for (;;) {
            auto step = job.next();
            if (step.layer || step.workspace || (step.state == StepState::complete && (step.value || step.graph)) ||
                (step.graph && step.value))
                throw std::logic_error("invalid decode job event");
            if (!step) break;
            if (step.state != StepState::advanced)
                throw std::logic_error("invalid decode job event");
            observed = true;
            if (step.graph) { capture.step(*step.graph); continue; }
            if (!step.value) continue;
            if (!result_required) throw std::logic_error("warmup returned a decode result");
            auto value = std::move(*step.value);
            capture.finish();
            job = {};
            return value;
        }
        if (result_required && observed) throw std::logic_error("decode job ended without a result");
        capture.finish();
        job = {};
        return {};
    } catch (...) {
        const auto failure = std::current_exception();
        try { capture.drain(); } catch (...) {}
        job = {};
        if (observed) try { state.release_graphs(); } catch (...) {}
        std::rethrow_exception(failure);
    }
}
} // namespace

std::optional<models::Decoded> decode(models::CausalModel& model, models::State& state,
        Executor& executor, weights::Cache& weights, const models::ForwardOptions& options,
        models::Input input, models::DecodeOptions decode_options) {
    StepSequence<models::Decoded> job;
    try { job = model.decode(std::move(input), state, options, decode_options); }
    catch (...) { Capture capture(executor, weights); try { capture.drain(); } catch (...) {} throw; }
    return drive(std::move(job), executor, weights, state, true);
}

void warmup(models::CausalModel& model, models::State& state, Executor& executor,
            weights::Cache& weights, models::DecodeOptions options) {
    StepSequence<models::Decoded> job;
    try { job = model.warmup(state, options); }
    catch (...) { Capture capture(executor, weights); try { capture.drain(); } catch (...) {} throw; }
    (void)drive(std::move(job), executor, weights, state, false);
}

std::int64_t decode_bucket(std::int64_t planned_len, std::int64_t context_capacity) {
    const std::int64_t quantum = planned_len <= 4096 ? 512 : (planned_len <= 16384 ? 1024 : 2048);
    return std::min(((planned_len + quantum - 1) / quantum) * quantum, context_capacity);
}

Decode::Decode(std::int64_t capacity, Device target, std::unique_ptr<Executor::Graph> execution_graph)
    : device(target), graph(std::move(execution_graph)), generated_capacity(std::max<std::int64_t>(capacity, 2048)) {}
void Decode::ensure_storage(std::int64_t vocabulary) {
    const auto i64 = TensorOptions().dtype(kInt64).device(device);
    if (!static_input) static_input = ops::empty({1, 1}, i64);
    if (!static_pos) static_pos = ops::empty({1}, i64);
    if (!static_len) static_len = ops::empty({1}, i64);
    if (!static_step) static_step = ops::empty({1}, i64);
    if (!generated) generated = ops::empty({generated_capacity}, i64);
    if (!random) random = ops::empty({1}, TensorOptions().dtype(kFloat32).device(device));
    if (!counts) counts = ops::empty({vocabulary}, TensorOptions().dtype(kInt32).device(device));
}
void Decode::initialize_inputs(const Tensor& pending, std::int64_t position, Tensor& random_host) {
    ops::copy_into(static_input, ops::reshape(pending, {1, 1}));
    auto first_generated = ops::slice(generated, 0, 0, 1);
    ops::copy_into(first_generated, ops::reshape(pending, {1}));
    ops::fill(static_pos, position);
    ops::fill(static_len, position + 1);
    ops::fill(static_step, 1);
    ops::fill(random_host, 0.5);
    upload_random(random_host);
    graph->wait();
}
void Decode::upload_random(const Tensor& random_host) { ops::copy_into(random, random_host); }
void Decode::capture_next(Tensor next) {
    static_next = std::move(next);
    ops::decode_commit(static_next, generated, static_step, static_input, static_pos, static_len);
}
bool Decode::matches(std::int64_t planned, const MfqSamplingParams& sampling, bool greedy,
                     std::span<const Tensor> owners) const {
    return valid_ && planned_len == planned && greedy_ == greedy &&
        (greedy || (temperature_ == sampling.temperature && top_k_ == sampling.top_k && top_p_ == sampling.top_p)) &&
        presence_ == sampling.presence_penalty && frequency_ == sampling.frequency_penalty &&
        repetition_ == sampling.repetition_penalty && graph->matches(owners);
}
void Decode::set_key(std::int64_t planned, const MfqSamplingParams& sampling, bool greedy) {
    planned_len = planned; greedy_ = greedy;
    temperature_ = sampling.temperature; top_k_ = sampling.top_k; top_p_ = sampling.top_p;
    presence_ = sampling.presence_penalty; frequency_ = sampling.frequency_penalty;
    repetition_ = sampling.repetition_penalty; valid_ = true;
}
void Decode::invalidate() {
    if (graph) graph->invalidate();
    static_next = {}; valid_ = false;
}
} // namespace tyloi::engine
#endif
