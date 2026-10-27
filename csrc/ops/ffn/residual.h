#pragma once

#include "native/cuda/profile.h"
#include "backend.h"

#include <array>

namespace mfq::cuda {

std::array<mfq_tensor_backend::Tensor, 2> full_norm(
    CudaProfiler& profiler, const mfq_tensor_backend::Tensor& ffn_norm, double eps, double offset,
    bool bf16, bool fused_norm, bool fp32, mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor output);

std::array<mfq_tensor_backend::Tensor, 2> linear_norm(
    CudaProfiler& profiler, const mfq_tensor_backend::Tensor& ffn_norm, double eps, double offset,
    bool fp32, mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor output);

mfq_tensor_backend::Tensor full_add(CudaProfiler& profiler, bool fp32, bool bf16_acc,
                mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor ff);

mfq_tensor_backend::Tensor linear_add(CudaProfiler& profiler, bool fp32, mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor ff);

} // namespace mfq::cuda
