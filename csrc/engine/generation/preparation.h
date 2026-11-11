#pragma once

#include "engine/generation/sequence.h"
#include "engine/models/input.h"

#include <memory>

namespace tyloi { class Executor; }
namespace tyloi::weights { class Cache; class Loan; }

namespace tyloi::engine {

/** Own a media preparation job and its synchronous workspace loans. */
class Preparation {
public:
    /** Take the job while borrowing the execution and weight owners through destruction. */
    Preparation(StepSequence<models::Prepared> job, Executor& executor, weights::Cache& weights);
    /** Drain pending work before retiring the job and any workspace loan. */
    ~Preparation();
    Preparation(const Preparation&) = delete;
    Preparation& operator=(const Preparation&) = delete;

    /** Advance until the next visible preparation result, retiring any completed loan. */
    StepResult<models::Prepared> next();

private:
    void close() noexcept;
    Executor& executor_;
    weights::Cache& weights_;
    std::unique_ptr<weights::Loan> loan_;
    StepSequence<models::Prepared> job_;
    bool closed_ = false;
};

} // namespace tyloi::engine
