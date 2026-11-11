#pragma once

#include "engine/generation/step.h"
#include "engine/models/causal_model.h"
#include "engine/generation/mtp/state.h"

#include <cstddef>
#include <cstdint>

namespace tyloi { class Executor; class ModelSource; }
namespace tyloi::weights { class Cache; }
namespace tyloi::engine { class Budget; }

namespace tyloi::engine::mtp {

/** Generate with the model's predictor and the request's owning state inside an execution scope. */
Generation generate(models::CausalModel& model, models::State& state, Executor& executor,
    Budget& budget, weights::Cache& weights, State& policy, const models::ForwardOptions& options,
    InferenceRequest& input, InferenceOutput& output, std::int64_t chunk,
    const models::Prepared* prepared, std::size_t reused, Tensor restored,
    Tensor* committed, std::size_t stable, const ModelSource* source, bool profile);

} // namespace tyloi::engine::mtp
