#pragma once

#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/native/context.h"

struct CudaScope {
    MfqCudaGuard device;
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::cuda::ContextGuard context;
#endif
    explicit CudaScope(
            int primary, [[maybe_unused]] tyloi::cuda::ContextSet& contexts)
        : device(primary)
#ifdef TYLOINFER_NATIVE_CUDA
        , context(contexts)
#endif
    {}
};
