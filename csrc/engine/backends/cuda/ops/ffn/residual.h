#pragma once

#include "engine/backends/cuda/native/profile.h"
#include "backend.h"

#include <array>

namespace tyloi::cuda {

std::array<::tyloi::Tensor, 2> full_norm(
    CudaProfiler& profiler, const ::tyloi::Tensor& ffn_norm, double eps, double offset,
    bool bf16, bool fused_norm, bool fp32, ::tyloi::Tensor residual, ::tyloi::Tensor output);

std::array<::tyloi::Tensor, 2> linear_norm(
    CudaProfiler& profiler, const ::tyloi::Tensor& ffn_norm, double eps, double offset,
    bool fp32, ::tyloi::Tensor residual, ::tyloi::Tensor output);

::tyloi::Tensor full_add(CudaProfiler& profiler, bool fp32, bool bf16_acc,
                ::tyloi::Tensor residual, ::tyloi::Tensor ff);

::tyloi::Tensor linear_add(CudaProfiler& profiler, bool fp32, ::tyloi::Tensor residual, ::tyloi::Tensor ff);

} // namespace tyloi::cuda
