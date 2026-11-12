#pragma once

#include "engine/models/batch.h"

namespace tyloi { class Executor; }
namespace tyloi::weights { class Cache; }
namespace tyloi::engine {
class Budget;

/** Reserve the combined per-device backing for a batch operation. */
void reserve_batch(Budget& budget, Executor& executor, const models::BatchGrowth& values,
    std::size_t margin, bool primary = false);

/** Drive model batch prefill events using the executor's working-set owner. */
std::vector<Tensor> prefill_batch(models::Batch& batch, Executor& executor, weights::Cache& weights,
    std::vector<models::BatchState*> states, std::vector<Tensor> ids);

/** Drive a batch graph plan through the shared capture coordinator. */
std::unique_ptr<models::Batch::Plan> plan_batch(models::Batch& batch, Executor& executor,
    weights::Cache& weights, const std::vector<models::BatchRow>& rows, bool greedy);

/** Own a controlled numerical batch and its independently admitted request states. */
class BatchDriver final {
public:
    BatchDriver(std::unique_ptr<models::Batch> batch, Executor& executor,
        weights::Cache& weights, Budget& budget);
    ~BatchDriver() noexcept;
    BatchDriver(const BatchDriver&) = delete;
    BatchDriver& operator=(const BatchDriver&) = delete;
    /** Admit and initialize batch-wide backing after the resource owner retains this driver. */
    void initialize();
    /** Admit and initialize a request state; this driver owns it until retirement. */
    models::BatchState& add(std::int64_t prompt_tokens);
    /** Run one full prompt span, preserving the production admission and event driver. */
    Tensor prefill(models::BatchState& state, Tensor ids);
    /** Run an eligible graph; an empty result means the model did not execute one. */
    std::optional<Tensor> capture(const std::vector<models::BatchRow>& rows,
        const Tensor& ids, const Tensor& host_ids, bool greedy);
    /** Retire graphs and states after the resource owner has drained. */
    void close();
private:
    Executor& executor_;
    weights::Cache& weights_;
    Budget& budget_;
    std::unique_ptr<models::Batch> batch_;
    std::vector<std::unique_ptr<models::BatchState>> states_;
};
}
