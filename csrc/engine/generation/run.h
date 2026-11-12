#pragma once

#include "engine/generation/step.h"

namespace tyloi { class Executor; class ModelSource; }
namespace tyloi::models { class CausalModel; class State; struct Info; struct ForwardOptions; struct Mtp; }
namespace tyloi::weights { class Cache; }
namespace tyloi::state { class Sessions; }
namespace tyloi::engine {
class Budget;
class Decode;
struct Config;
namespace mtp { struct State; }

/** Borrow one request's numerical and execution owners for the full generation lifetime. */
struct Run {
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
    mtp::State& mtp_state;
    /** The loaded predictor selected for this request, or null for the ordinary oracle. */
    models::Mtp* mtp = nullptr;
};

/** Create the shared request driver; resume and retire it in its owner's execution scope. */
Generation generate(Run run, InferenceRequest& request, InferenceOutput& output,
    RequestId id = {});
}
