#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cstdint>

namespace mfq_jit_backward {
/** Launch a supported packed-backward scalar conversion on the supplied stream. */
template<class Destination, class Source>
void launch_cast(const Source* source, Destination* destination, std::int64_t count, cudaStream_t stream);
template<> void launch_cast<__half, __nv_bfloat16>(const __nv_bfloat16*, __half*, std::int64_t, cudaStream_t);
template<> void launch_cast<__nv_bfloat16, __half>(const __half*, __nv_bfloat16*, std::int64_t, cudaStream_t);
template<> void launch_cast<float, __half>(const __half*, float*, std::int64_t, cudaStream_t);
/** Sum float partials into a half destination. */
void launch_split_float_reduce_to_half(const float*, __half*, int rows, int width, int splits, cudaStream_t);
/** Sum half partials into a half destination. */
void launch_split_half_reduce_to_half(const __half*, __half*, int rows, int width, int splits, cudaStream_t);
/** Run the original half GEMM using the supplied cuBLAS handle. */
cublasStatus_t raw_half_gemm_nn(cublasHandle_t, const __half*, const __half*, __half*, int rows, int outputs, int width);
/** Run the original float GEMM using the supplied cuBLAS handle. */
cublasStatus_t raw_float_gemm_nn(cublasHandle_t, const float*, const float*, float*, int rows, int outputs, int width);
}
