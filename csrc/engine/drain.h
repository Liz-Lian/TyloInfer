#pragma once

#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/tq.h"
#include "engine/backends/cuda/native/branches.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/state/kv_offload.h"
#include "engine/weights/cache.h"
#include "engine/weights/placement.h"

#include <exception>
#include <unordered_set>

namespace tyloi::cuda {

inline void drain_cuda(
        Executor& executor, const ::tyloi::weights::Cache& weights, KvOffloadStore* offload,
        const LayerPlacementConfig& layer_placement,
        [[maybe_unused]] const ParallelConfig& tensor_parallel,
        [[maybe_unused]] const ParallelConfig& expert_parallel) {
    if (executor.contexts().empty() && weights.empty() && !offload &&
        executor.tq()->empty() && executor.branches().empty() &&
        executor.collectives().streams().empty()) return;
    auto scope = executor.enter(layer_placement.primary_device());
    std::exception_ptr failure;
    const auto finish = [&](auto&& work) {
        try { work(); } catch (...) { if (!failure) failure = std::current_exception(); }
    };
    finish([&] { weights.drain(); });
    finish([&] { static_cast<tyloi::Executor&>(executor).drain(); });
    if (failure) std::rethrow_exception(failure);
}

} // namespace tyloi::cuda
