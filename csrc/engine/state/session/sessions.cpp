#include "engine/state/session/sessions.h"

#include "engine/budget.h"
#include "engine/executor.h"
#include "engine/generation/prefill_budget.h"
#include "engine/models/common/mtp.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/state.h"
#include "engine/weights/cache.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tyloi::state {
namespace {

void drain(Executor& executor, weights::Cache& weights) {
    std::exception_ptr failure;
    try { weights.drain(); } catch (...) { failure = std::current_exception(); }
    try { executor.drain(); } catch (...) { if (!failure) failure = std::current_exception(); }
    if (failure) std::rethrow_exception(failure);
}

} // namespace

void restore(models::State& state, const TextSessionState& snapshot, Executor& executor,
        weights::Cache& weights, engine::Budget& budget, models::Mtp* mtp) {
    const TextSessionState held = snapshot;
    std::unique_ptr<models::Restore> target, predictor;
    try {
        if (mtp) {
            if (!mtp->supports_session_state() || !state.predictor ||
                !state.predictor->belongs_to(*mtp) || !held.mtp || held.cache_pos <= 0 ||
                !mtp->session_boundary(held.cache_pos, held.mtp->cache_pos))
                throw std::invalid_argument("joint predictor session state is incompatible");
        }
        target = state.prepare_restore(held);
        if (mtp) {
            predictor = mtp->prepare_restore(*state.predictor, *held.mtp);
            if (!target || !predictor)
                throw std::invalid_argument("joint predictor session restore is unsupported");
        }
        std::vector<std::pair<Device, std::size_t>> growth;
        const auto merge = [&](const auto& values) {
            for (const auto& [device, bytes] : values) {
                auto found = std::find_if(growth.begin(), growth.end(),
                    [device](const auto& value) { return value.first == device; });
                if (found == growth.end()) growth.emplace_back(device, bytes);
                else {
                    if (found->second > std::numeric_limits<std::size_t>::max() - bytes)
                        throw std::overflow_error("session restore allocation size overflow");
                    found->second += bytes;
                }
            }
        };
        merge(target ? target->growth() : state.restore_growth(held));
        if (predictor) merge(predictor->growth());
        for (const auto& [device, bytes] : growth)
            if (bytes) budget.reserve_kv(device, engine::prefill_budget_add(bytes, std::size_t(64) << 20));
        drain(executor, weights);
        if (target) target->stage();
        else state.restore_text_session_state(held);
        if (predictor) predictor->stage();
        drain(executor, weights);
        if (target) target->publish();
        if (predictor) predictor->publish();
    } catch (...) {
        const auto error = std::current_exception();
        try { drain(executor, weights); } catch (...) {}
        std::rethrow_exception(error);
    }
}

} // namespace tyloi::state
