#pragma once
#include "tyloinfer/gguf_quant.h"
#include <cuda_runtime_api.h>
#include <cstdint>
namespace mfq::cuda::gguf {
// Decode an arbitrary row view directly; optional indices select embedding
// rows. No intermediate compressed image or permanent native copy is needed.
void decode_quad_rows(ggml_type,const void* packed,std::int64_t storage_rows,
    std::int64_t columns,std::int64_t begin,std::int64_t rows,
    const std::int64_t* indices,std::int64_t count,void* output,
    bool half_output,cudaStream_t stream);
void decode_quad_rows_scaled(ggml_type type,const void* packed,std::int64_t storage_rows,
    std::int64_t columns,std::int64_t begin,std::int64_t rows,
    const std::int32_t* exponents,void* output,cudaStream_t stream);
}
