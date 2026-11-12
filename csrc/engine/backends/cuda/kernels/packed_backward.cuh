#pragma once

#include "packed_backward_raw.cuh"

// Retained only for MFQ backward callers not yet migrated to the Python adapter.
namespace mfq_packed_backward {
namespace raw = ::tyloi::cuda::backward_raw;
using raw::launch_cast;
using raw::launch_split_float_reduce_to_half;
using raw::launch_split_half_reduce_to_half;

/** Run the original FP16 backward GEMM on the supplied stream. */
inline void launch_half_gemm_nn(const __half* gradient, const __half* weight,
        __half* output, int rows, int outputs, int width, cudaStream_t stream) {
    const auto handle = mfq_current_cublas_handle();
    TYLOINFER_RUNTIME_CHECK(cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS,
        "packed backward cublasSetStream failed");
    TYLOINFER_RUNTIME_CHECK(raw::half_gemm_nn(handle, gradient, weight, output, rows, outputs, width) == CUBLAS_STATUS_SUCCESS,
        "packed backward GEMM failed");
}
/** Run the original FP32 backward GEMM on the supplied stream. */
inline void launch_float_gemm_nn(const float* gradient, const float* weight,
        float* output, int rows, int outputs, int width, cudaStream_t stream) {
    const auto handle = mfq_current_cublas_handle();
    TYLOINFER_RUNTIME_CHECK(cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS,
        "packed backward cublasSetStream failed");
    TYLOINFER_RUNTIME_CHECK(raw::float_gemm_nn(handle, gradient, weight, output, rows, outputs, width) == CUBLAS_STATUS_SUCCESS,
        "packed backward FP32 GEMM failed");
}

/** Compute input gradients with the original dtype conversions and GEMM accumulation. */
inline void launch_dense_half_weight(
        const ::tyloi::DenseTensor & output_gradient,
        const ::tyloi::DenseTensor & half_weight,
        ::tyloi::DenseTensor & input_gradient,
        int rows,
        int outputs,
        int width,
        cudaStream_t stream) {
    const auto dtype = output_gradient.scalar_type();
    if (dtype == ::tyloi::kFloat16) {
        launch_half_gemm_nn(
            reinterpret_cast<const __half *>(
                output_gradient.data_ptr<mfq_half>()),
            reinterpret_cast<const __half *>(half_weight.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(input_gradient.data_ptr<mfq_half>()),
            rows, outputs, width, stream);
        return;
    }
    if (dtype == ::tyloi::kBFloat16) {
        auto half_gradient = ::tyloi::cuda::empty(
            {rows, outputs}, half_weight.options());
        auto half_result = ::tyloi::cuda::empty(
            {rows, width}, half_weight.options());
        launch_cast<__half, __nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16 *>(
                output_gradient.data_ptr<mfq_bfloat16>()),
            reinterpret_cast<__half *>(half_gradient.data_ptr<mfq_half>()),
            static_cast<int64_t>(rows) * outputs, stream);
        launch_half_gemm_nn(
            reinterpret_cast<const __half *>(half_gradient.data_ptr<mfq_half>()),
            reinterpret_cast<const __half *>(half_weight.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(half_result.data_ptr<mfq_half>()),
            rows, outputs, width, stream);
        launch_cast<__nv_bfloat16, __half>(
            reinterpret_cast<const __half *>(half_result.data_ptr<mfq_half>()),
            reinterpret_cast<__nv_bfloat16 *>(
                input_gradient.data_ptr<mfq_bfloat16>()),
            static_cast<int64_t>(rows) * width, stream);
        return;
    }
    auto float_weight = ::tyloi::cuda::empty(
        {outputs, width}, half_weight.options().dtype(
            ::tyloi::kFloat32));
    launch_cast<float, __half>(
        reinterpret_cast<const __half *>(half_weight.data_ptr<mfq_half>()),
        float_weight.data_ptr<float>(),
        static_cast<int64_t>(outputs) * width, stream);
    launch_float_gemm_nn(
        output_gradient.data_ptr<float>(), float_weight.data_ptr<float>(),
        input_gradient.data_ptr<float>(), rows, outputs, width, stream);
}

} // namespace mfq_packed_backward
