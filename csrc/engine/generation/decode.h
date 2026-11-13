#pragma once

#include "engine/executor.h"
#include "engine/ops.h"
#include "engine/requests/sampling.h"
#include "engine/models/input.h"
#include "engine/generation/sequence.h"
#include "engine/weights/cache.h"
#include <optional>

namespace tyloi::weights { class Cache; }
namespace tyloi::models { class CausalModel; class State; }

namespace tyloi::engine {

/** Coordinate graph phases and synthetic warmup resources in the runtime. */
class Capture {
public:
    /** Borrow resource owners that outlive this coordinator and its numerical jobs. */
    Capture(Executor& executor, weights::Cache& weights);
    /** Restore any remaining warmup policy after the caller retires its job. */
    ~Capture();
    /** Apply one model-reported graph boundary. */
    void step(GraphPhase phase);
    /** Release warmup allocations and restore the prior cache policy. */
    void finish();
    /** Drain both resource owners, propagating the first failure. */
    void drain();
private:
    Executor& executor_;
    weights::Cache& weights_;
    std::unique_ptr<weights::Cache::Warmup> warmup_;
    bool target_ = false;
};

/** Drive model-owned decode phases and return no value when ordinary forward is needed. */
std::optional<models::Decoded> decode(models::CausalModel& model, models::State& state,
    Executor& executor, weights::Cache& weights, const models::ForwardOptions& options,
    models::Input input, models::DecodeOptions decode_options);
/** Drive optional model warmup phases within the execution owner. */
void warmup(models::CausalModel& model, models::State& state, Executor& executor,
    weights::Cache& weights, models::DecodeOptions options);

std::int64_t decode_bucket(std::int64_t planned_len, std::int64_t context_capacity);

class Decode {
public:
    Decode(std::int64_t capacity, Device device, std::unique_ptr<Executor::Graph> graph);
    void ensure_storage(std::int64_t vocabulary);
    void initialize_inputs(const Tensor& pending, std::int64_t position, Tensor& random_host);
    void upload_random(const Tensor& random_host);
    void capture_next(Tensor next);
    bool matches(std::int64_t planned, const MfqSamplingParams& sampling, bool greedy,
                 std::span<const Tensor> owners) const;
    void set_key(std::int64_t planned, const MfqSamplingParams& sampling, bool greedy);
    void invalidate();

    Device device;
    Tensor static_input, static_pos, static_len, static_step;
    Tensor generated, random, counts, static_next;
    std::unique_ptr<Executor::Graph> graph;
    std::int64_t generated_capacity;
    std::uint64_t captures = 0, reuses = 0;
private:
    std::int64_t planned_len = 0;
    double temperature_ = 0, top_p_ = 1, presence_ = 0, frequency_ = 0, repetition_ = 1;
    std::int32_t top_k_ = 0;
    bool greedy_ = false, valid_ = false;
};
} // namespace tyloi::engine
