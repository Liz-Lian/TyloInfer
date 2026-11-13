#pragma once

#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/native/context.h"

struct CudaScope {
    MfqCudaGuard device;
    tyloi::cuda::ContextGuard context;
    explicit CudaScope(
            int primary, [[maybe_unused]] tyloi::cuda::ContextSet& contexts)
        : device(primary)
        , context(contexts)
    {}
};
