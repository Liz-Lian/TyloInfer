#pragma once

#include <memory>

namespace tyloi { class Executor; }
namespace tyloi::weights { class Cache; }
namespace tyloi::models { class CausalModel; class State; }
namespace tyloi::engine {
class Batch;
class Budget;
struct Config;

/** Construct the shared scheduler around model-owned batch computation. */
std::unique_ptr<Batch> make_batch(models::CausalModel& model, models::State* serial, Executor& executor,
    weights::Cache& weights, Budget& budget, const Config& config);
} // namespace tyloi::engine
