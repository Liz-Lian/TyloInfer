#pragma once

#include "engine/backends/cuda/native/profile.h"
#include "backend.h"

#include <array>

namespace tyloi::cuda {

std::array<::tyloi::DenseTensor, 2> full_norm(
    CudaProfiler& profiler,
#ifdef TYLOINFER_NATIVE_CUDA
    const ::tyloi::Tensor& ffn_norm,
#else
    const ::tyloi::DenseTensor& ffn_norm,
#endif
    double eps, double offset,
    bool bf16, bool fused_norm, bool fp32, ::tyloi::DenseTensor residual, ::tyloi::DenseTensor output);

std::array<::tyloi::DenseTensor, 2> linear_norm(
    CudaProfiler& profiler,
#ifdef TYLOINFER_NATIVE_CUDA
    const ::tyloi::Tensor& ffn_norm,
#else
    const ::tyloi::DenseTensor& ffn_norm,
#endif
    double eps, double offset,
    bool fp32, ::tyloi::DenseTensor residual, ::tyloi::DenseTensor output);

::tyloi::DenseTensor full_add(CudaProfiler& profiler, bool fp32, bool bf16_acc,
                ::tyloi::DenseTensor residual, ::tyloi::DenseTensor ff);

::tyloi::DenseTensor linear_add(CudaProfiler& profiler, bool fp32, ::tyloi::DenseTensor residual, ::tyloi::DenseTensor ff);

} // namespace tyloi::cuda
