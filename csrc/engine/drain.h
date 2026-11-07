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
#ifdef TYLOINFER_NATIVE_CUDA
    if (executor.contexts().empty() && weights.empty() && !offload &&
        executor.tq()->empty() && executor.pipelines().empty() && executor.branches().empty() &&
        executor.collectives().streams().empty()) return;
#endif
    auto scope = executor.enter(layer_placement.primary_device());
    std::exception_ptr failure;
    const auto finish = [&](auto&& work) {
        try { work(); } catch (...) { if (!failure) failure = std::current_exception(); }
    };
    finish([&] { weights.drain(); });
#ifdef TYLOINFER_NATIVE_CUDA
    if (!offload) {
        finish([&] { static_cast<tyloi::Executor&>(executor).drain(); });
    } else
#endif
    {
        finish([&] { executor.join(); });
        if (offload) finish([&] { offload->flush(); });
#ifdef TYLOINFER_NATIVE_CUDA
        finish([&] { executor.wait(); });
#else
        std::unordered_set<int> devices{layer_placement.primary_device()};
        for (const auto* group : {&tensor_parallel.devices, &expert_parallel.devices, &layer_placement.devices})
            devices.insert(group->begin(), group->end());
        finish([&] { executor.wait(devices); });
#endif
    }
    if (failure) std::rethrow_exception(failure);
}

} // namespace tyloi::cuda
