#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cstdint>

namespace tyloi::cuda::backward_raw {
/** Multiply row-major half gradients and half weights using the caller's configured handle. */
inline cublasStatus_t half_gemm_nn(cublasHandle_t handle,
        const __half* output_gradient, const __half* weight, __half* input_gradient,
        int rows, int outputs, int width) {
    const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    return cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        width, rows, outputs, &alpha, weight, CUDA_R_16F, width,
        output_gradient, CUDA_R_16F, outputs, &beta, input_gradient, CUDA_R_16F,
        width, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

template <typename Destination, typename Source>
__device__ __forceinline__ Destination cast_value(Source value) {
    return static_cast<Destination>(value);
}

template <>
__device__ __forceinline__ __half cast_value<__half, __nv_bfloat16>(
        __nv_bfloat16 value) {
    return __float2half_rn(__bfloat162float(value));
}

template <>
__device__ __forceinline__ __nv_bfloat16 cast_value<__nv_bfloat16, __half>(
        __half value) {
    return __float2bfloat16_rn(__half2float(value));
}

template <>
__device__ __forceinline__ float cast_value<float, __half>(__half value) {
    return __half2float(value);
}

template <typename Destination, typename Source>
__global__ void cast_kernel(
        const Source * __restrict__ source,
        Destination * __restrict__ destination,
        int64_t count) {
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < count;
         index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        destination[index] = cast_value<Destination>(source[index]);
    }
}

/** Convert contiguous gradient storage using the original element conversion. */
template <typename Destination, typename Source>
inline void launch_cast(
        const Source * source,
        Destination * destination,
        int64_t count,
        cudaStream_t stream) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<int64_t>(
        (count + threads - 1) / threads, 65535));
    cast_kernel<Destination, Source><<<blocks, threads, 0, stream>>>(
        source, destination, count);
}

static __global__ void split_float_reduce_to_half_kernel(
        const float * __restrict__ partials,
        __half * __restrict__ destination,
        int rows,
        int width,
        int splits) {
    const int64_t total = static_cast<int64_t>(rows) * width;
    for (int64_t logical = static_cast<int64_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         logical < total;
         logical += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(logical / width);
        const int column = static_cast<int>(logical -
            static_cast<int64_t>(row) * width);
        float accumulator = 0.0f;
        for (int split = 0; split < splits; ++split) {
            accumulator += partials[
                (static_cast<int64_t>(split) * rows + row) * width + column];
        }
        destination[logical] = __float2half_rn(accumulator);
    }
}

/** Sum split-major F32 partials in split order before rounding once to F16. */
inline void launch_split_float_reduce_to_half(
        const float * partials,
        __half * destination,
        int rows,
        int width,
        int splits,
        cudaStream_t stream) {
    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(rows) * width;
    const int blocks = static_cast<int>(std::min<int64_t>(
        (total + threads - 1) / threads, 65535));
    split_float_reduce_to_half_kernel<<<blocks, threads, 0, stream>>>(
        partials, destination, rows, width, splits);
}

static __global__ void split_half_reduce_to_half_kernel(
        const __half * __restrict__ partials,
        __half * __restrict__ destination,
        int64_t total,
        int splits) {
    for (int64_t logical = static_cast<int64_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         logical < total;
         logical += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        float accumulator = 0.0f;
        for (int split = 0; split < splits; ++split) {
            accumulator += __half2float(partials[
                static_cast<int64_t>(split) * total + logical]);
        }
        destination[logical] = __float2half_rn(accumulator);
    }
}

static __global__ void split_half2_reduce_to_half_kernel(
        const __half2 * __restrict__ partials,
        __half2 * __restrict__ destination,
        int64_t pairs,
        int splits) {
    for (int64_t logical = static_cast<int64_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         logical < pairs;
         logical += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        float2 accumulator = make_float2(0.0f, 0.0f);
        for (int split = 0; split < splits; ++split) {
            const __half2 value = partials[
                static_cast<int64_t>(split) * pairs + logical];
            const float2 unpacked = __half22float2(value);
            accumulator.x += unpacked.x;
            accumulator.y += unpacked.y;
        }
        destination[logical] = __floats2half2_rn(
            accumulator.x, accumulator.y);
    }
}

/** Sum split-major F16 partials in F32 and preserve the original half2 path. */
inline void launch_split_half_reduce_to_half(
        const __half * partials,
        __half * destination,
        int rows,
        int width,
        int splits,
        cudaStream_t stream) {
    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(rows) * width;
    if ((total & 1) == 0) {
        const int64_t pairs = total / 2;
        const int blocks = static_cast<int>(std::min<int64_t>(
            (pairs + threads - 1) / threads, 65535));
        split_half2_reduce_to_half_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __half2 *>(partials),
            reinterpret_cast<__half2 *>(destination), pairs, splits);
        return;
    }
    const int blocks = static_cast<int>(std::min<int64_t>(
        (total + threads - 1) / threads, 65535));
    split_half_reduce_to_half_kernel<<<blocks, threads, 0, stream>>>(
        partials, destination, total, splits);
}

/** Multiply row-major float gradients and float weights using the caller's configured handle. */
inline cublasStatus_t float_gemm_nn(cublasHandle_t handle,
        const float* output_gradient, const float* weight, float* input_gradient,
        int rows, int outputs, int width) {
    const float alpha = 1.0f, beta = 0.0f;
    return cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        width, rows, outputs, &alpha, weight, CUDA_R_32F, width,
        output_gradient, CUDA_R_32F, outputs, &beta, input_gradient, CUDA_R_32F,
        width, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
} // namespace tyloi::cuda::backward_raw
