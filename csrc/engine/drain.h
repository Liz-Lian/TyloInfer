#pragma once

#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/tq.h"
#include "engine/backends/cuda/native/branches.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/state/kv_offload.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/weights/placement.h"
#include "engine/weights/routed_cache.h"

#include <exception>
#include <unordered_set>

namespace mfq::cuda {

inline void drain_cuda(
        Executor& executor, const RoutedCache& weights, KvOffloadStore* offload,
        const LayerPlacementConfig& layer_placement,
        [[maybe_unused]] const ParallelConfig& tensor_parallel,
        [[maybe_unused]] const ParallelConfig& expert_parallel) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    if (executor.contexts().empty() && !weights.moe &&
        !weights.tq && !offload &&
        executor.tq()->empty() && executor.pipelines().empty() && executor.branches().empty() &&
        executor.collectives().streams().empty()) return;
#endif
    auto scope = executor.enter(layer_placement.primary_device());
    std::exception_ptr failure;
    const auto finish = [&](auto&& work) {
        try { work(); } catch (...) { if (!failure) failure = std::current_exception(); }
    };
    finish([&] { drain_moe_expert_cache(weights.moe); });
    finish([&] { executor.pipelines().drain(); });
    finish([&] { executor.collectives().drain(); });
    if (offload) finish([&] { offload->flush(); });
    // shortcut: retirement fences the whole device; use events when every consumer has a completion handle.
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    for (const auto& [device, context] : executor.contexts()) finish([&] {
        MfqCudaGuard guard(device);
        TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());
    });
#else
    std::unordered_set<int> devices{layer_placement.primary_device()};
    for (const auto* group : {&tensor_parallel.devices, &expert_parallel.devices, &layer_placement.devices})
        devices.insert(group->begin(), group->end());
    for (int device : devices) finish([&] {
        MfqCudaGuard guard(device);
        TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());
    });
#endif
    finish([&] { executor.tq()->drain(); });
    finish([&] { executor.branches().drain(); });
    if (failure) std::rethrow_exception(failure);
}

} // namespace mfq::cuda
