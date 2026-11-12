#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "packed_backward_raw.cuh"

namespace mfq_jit_backward {
namespace raw = ::tyloi::cuda::backward_raw;
using raw::launch_cast;
using raw::launch_split_float_reduce_to_half;
using raw::launch_split_half_reduce_to_half;

/** Run the original FP16 backward GEMM on the supplied stream. */
inline void launch_half_gemm_nn(const __half* gradient, const __half* weight,
        __half* output, int rows, int outputs, int width, cudaStream_t stream) {
    const auto handle = at::cuda::getCurrentCUDABlasHandle();
    TORCH_CHECK(cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS,
        "packed backward cublasSetStream failed");
    TORCH_CHECK(raw::half_gemm_nn(handle, gradient, weight, output, rows, outputs, width) == CUBLAS_STATUS_SUCCESS,
        "packed backward GEMM failed");
}
/** Run the original FP32 backward GEMM on the supplied stream. */
inline void launch_float_gemm_nn(const float* gradient, const float* weight,
        float* output, int rows, int outputs, int width, cudaStream_t stream) {
    const auto handle = at::cuda::getCurrentCUDABlasHandle();
    TORCH_CHECK(cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS,
        "packed backward cublasSetStream failed");
    TORCH_CHECK(raw::float_gemm_nn(handle, gradient, weight, output, rows, outputs, width) == CUBLAS_STATUS_SUCCESS,
        "packed backward FP32 GEMM failed");
}

/** Compute input gradients with the original dtype conversions and GEMM accumulation. */
inline void launch_dense_half_weight(
        const torch::Tensor & output_gradient,
        const torch::Tensor & half_weight,
        torch::Tensor & input_gradient,
        int rows,
        int outputs,
        int width,
        cudaStream_t stream) {
    const auto dtype = output_gradient.scalar_type();
    if (dtype == at::kHalf) {
        launch_half_gemm_nn(
            reinterpret_cast<const __half *>(
                output_gradient.data_ptr<at::Half>()),
            reinterpret_cast<const __half *>(half_weight.data_ptr<at::Half>()),
            reinterpret_cast<__half *>(input_gradient.data_ptr<at::Half>()),
            rows, outputs, width, stream);
        return;
    }
    if (dtype == at::kBFloat16) {
        auto half_gradient = torch::empty(
            {rows, outputs}, half_weight.options());
        auto half_result = torch::empty(
            {rows, width}, half_weight.options());
        launch_cast<__half, __nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16 *>(
                output_gradient.data_ptr<at::BFloat16>()),
            reinterpret_cast<__half *>(half_gradient.data_ptr<at::Half>()),
            static_cast<int64_t>(rows) * outputs, stream);
        launch_half_gemm_nn(
            reinterpret_cast<const __half *>(half_gradient.data_ptr<at::Half>()),
            reinterpret_cast<const __half *>(half_weight.data_ptr<at::Half>()),
            reinterpret_cast<__half *>(half_result.data_ptr<at::Half>()),
            rows, outputs, width, stream);
        launch_cast<__nv_bfloat16, __half>(
            reinterpret_cast<const __half *>(half_result.data_ptr<at::Half>()),
            reinterpret_cast<__nv_bfloat16 *>(
                input_gradient.data_ptr<at::BFloat16>()),
            static_cast<int64_t>(rows) * width, stream);
        return;
    }
    auto float_weight = torch::empty(
        {outputs, width}, half_weight.options().dtype(
            at::kFloat));
    launch_cast<float, __half>(
        reinterpret_cast<const __half *>(half_weight.data_ptr<at::Half>()),
        float_weight.data_ptr<float>(),
        static_cast<int64_t>(outputs) * width, stream);
    launch_float_gemm_nn(
        output_gradient.data_ptr<float>(), float_weight.data_ptr<float>(),
        input_gradient.data_ptr<float>(), rows, outputs, width, stream);
}

} // namespace mfq_jit_backward
