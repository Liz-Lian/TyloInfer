#pragma once
#include "numeric_type.h"
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cstddef>
#include <cstdint>

namespace tyloi::cuda::fp8_sq_raw {
/** Describe the validated FP8-SQ packed layout. */
struct Layout {
    int outputs=0,width=0,block_rows=0,block_columns=0,scale_rows=0,scale_columns=0,scale_kind=0;
    std::size_t palettes=0,symbols=0,scales=0;
};
/** Decode MXFP8-SQ or FP8-128SQ into F16 or F32. */
void dequant(const std::uint8_t*,const std::uint8_t*,const std::int32_t*,void*,Layout,bool,bool,cudaStream_t);
/** Apply the original direct packed projection. */
void matmul(const std::uint8_t*,const std::uint8_t*,const std::int32_t*,const void*,void*,Layout,int,bool,NumericType,cudaStream_t);
/** Apply routed packed projection into caller-owned output. */
void routed(const std::uint8_t*,const std::uint8_t*,const std::int32_t*,const void*,void*,const std::int32_t*,const std::int32_t*,Layout,int,int,int,int,int,bool,bool,cudaStream_t);
/** Multiply by the decoded row-major weight with the original cuBLAS policy. */
cublasStatus_t gemm(cublasHandle_t,const void*,const void*,void*,int,int,int,NumericType,cudaStream_t);
}
