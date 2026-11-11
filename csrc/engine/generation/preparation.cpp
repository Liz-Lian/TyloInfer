#include "engine/generation/preparation.h"

#include "engine/executor.h"
#include "engine/weights/cache.h"
#include "engine/weights/workspace.h"

#include <exception>
#include <stdexcept>
#include <utility>

namespace tyloi::engine {

Preparation::Preparation(StepSequence<models::Prepared> job, Executor& executor, weights::Cache& weights)
    : executor_(executor), weights_(weights), job_(std::move(job)) {}

Preparation::~Preparation() { close(); }

void Preparation::close() noexcept {
    if (closed_) return;
    closed_ = true;
    try { weights_.drain(); } catch (...) {}
    try { executor_.drain(); } catch (...) {}
    job_ = {};
    loan_.reset();
}

StepResult<models::Prepared> Preparation::next() {
    if (closed_) return {};
    StepResult<models::Prepared> step;
    try {
        for (;;) {
            step = job_.next();
            if (step.graph) throw std::logic_error("preparation does not support graph events");
            if (!step.workspace) {
                if (loan_) throw std::logic_error("preparation yielded while workspace is borrowed");
                return step;
            }
            if (step.state != StepState::advanced || step.value)
                throw std::logic_error("invalid preparation workspace event");
            const auto request = *step.workspace;
            if (request.tile || request.extra)
                throw std::logic_error("preparation does not support tiled or multiple-output workspace");
            if (request.bytes) {
                if (loan_ || request.output) throw std::logic_error("invalid preparation workspace acquisition");
                loan_ = std::make_unique<weights::Loan>(weights_);
                loan_->begin(request.bytes);
            } else {
                if (!loan_ || !request.output) throw std::logic_error("invalid preparation workspace completion");
                loan_->finish(*request.output);
                loan_.reset();
            }
        }
    } catch (...) {
        const auto error = std::current_exception();
        close();
        std::rethrow_exception(error);
    }
}

} // namespace tyloi::engine
