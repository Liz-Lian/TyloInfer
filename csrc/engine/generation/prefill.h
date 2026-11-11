#pragma once

#include "engine/models/input.h"
#include "engine/models/metadata.h"

namespace tyloi { class Executor; }
namespace tyloi::weights { class Cache; }
namespace tyloi::models { class CausalModel; class State; }
namespace tyloi::engine { class Budget; class InferenceOutput; }

namespace tyloi::engine {

/** Drive a model's prefill job or its ordinary forward path with allocation reservations. */
models::Prefill prefill(models::CausalModel& model, models::State& state, Executor& executor,
    weights::Cache& weights, Budget& budget, const models::ForwardOptions& options,
    models::Input input, std::int64_t compute, InferenceOutput& output);

} // namespace tyloi::engine
