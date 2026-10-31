#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <mutex>
#include <stdexcept>

namespace mfq::cuda::hadamard {
// Larger transforms use one CTA's dynamic shared memory. Query the actual
// per-device/function limits, and only ever raise the function's opt-in cap.
inline void prepare_shared(const void* function, std::size_t bytes) {
    int device = 0, regular = 0;
    auto check = [](cudaError_t status) {
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    };
    check(cudaGetDevice(&device));
    check(cudaDeviceGetAttribute(&regular, cudaDevAttrMaxSharedMemoryPerBlock, device));
    if (bytes <= std::size_t(regular)) return;
    int optin = 0;
    check(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    if (bytes > std::size_t(std::max(regular, optin)))
        throw std::invalid_argument("Hadamard block exceeds CUDA shared-memory capacity");
    static std::mutex mutex;
    std::lock_guard lock(mutex);
    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes, function));
    if (bytes > std::size_t(attributes.maxDynamicSharedSizeBytes))
        check(cudaFuncSetAttribute(function, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)));
}
}
