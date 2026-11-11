#include "engine/generation/prefill.h"

#include "engine/budget.h"
#include "engine/executor.h"
#include "engine/generation/prefill_budget.h"
#include "engine/models/causal_model.h"
#include "engine/requests/output.h"
#include "engine/weights/cache.h"

#include <exception>
#include <optional>
#include <stdexcept>
#include <utility>

namespace tyloi::engine {

std::int64_t prefill_span(const models::Info& info, std::int64_t tokens,
    std::int64_t compute, bool transformed, bool retained) {
    return info.prefill == models::PrefillPolicy::merged && !transformed && !retained ? tokens : compute;
}

models::Prefill prefill(models::CausalModel& model, models::State& state, Executor& executor,
    weights::Cache& weights, Budget& budget, const models::ForwardOptions& options,
    models::Input input, std::int64_t compute, InferenceOutput& output) {
    tyloi::StepSequence<models::Prefill> job;
    std::optional<weights::Cache::Priority> priority;
    models::Prefill result;
    bool observed = false;
    bool computed = false;
    auto timer = executor.timer();
    const auto drain = [&] {
        std::exception_ptr failure;
        try { weights.drain(); } catch (...) { failure = std::current_exception(); }
        try { executor.drain(); } catch (...) { if (!failure) failure = std::current_exception(); }
        if (failure) std::rethrow_exception(failure);
    };
    try {
        for (const auto& [device, bytes] : state.reset_growth(input.ids.size(0)))
            if (bytes) budget.reserve_kv(device, prefill_budget_add(bytes, std::size_t(64) << 20));
        for (const auto& [device, bytes] : state.growth(input.ids.size(1), options, input.confirmed))
            if (bytes) budget.reserve_kv(device, prefill_budget_add(bytes, std::size_t(64) << 20));
        job = model.prefill(input, state, options, compute);
        while (!output.stopped()) {
            auto step = job.next();
            if (!step) {
                if (observed) throw std::runtime_error("prefill job ended without a result");
                auto* raw = input.raw_hidden;
                result.hidden = model.forward(std::move(input), state, options);
                if (raw) result.raw = *raw;
                return result;
            }
            observed = true;
            if (!step.value) continue;
            result = std::move(*step.value);
            if (result.phase == models::Prefill::Phase::computed) {
                timer->stop();
                computed = true;
            } else if (result.phase == models::Prefill::Phase::encode) {
                if (result.first < result.last) {
                    if (priority) { priority->finish(); priority.reset(); }
                    priority.emplace(weights, result.first, result.last);
                }
            } else {
                if (priority) { priority->finish(); priority.reset(); }
                if (result.phase == models::Prefill::Phase::complete) {
                    if (computed) result.elapsed_ms = timer->elapsed_ms();
                    return result;
                }
            }
        }
        drain();
        return {};
    } catch (...) {
        auto failure = std::current_exception();
        try { drain(); } catch (...) {}
        std::rethrow_exception(failure);
    }
}

} // namespace tyloi::engine
