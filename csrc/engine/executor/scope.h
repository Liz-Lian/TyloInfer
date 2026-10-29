#pragma once

#include "native/cuda/tensor/backend.h"
#include "native/cuda/context.h"

struct CudaScope {
    MfqCudaGuard device;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    mfq::cuda::ContextGuard context;
#endif
    explicit CudaScope(
            int primary, [[maybe_unused]] mfq::cuda::ContextSet& contexts)
        : device(primary)
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        , context(contexts)
#endif
    {}
};
