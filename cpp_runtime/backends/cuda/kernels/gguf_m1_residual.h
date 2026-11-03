#pragma once
#include "tyloinfer/gguf_quant.h"
#include <cuda_runtime_api.h>
#include <cstdint>

namespace mfq::cuda::gguf {
bool supports_m1_residual(ggml_type type) noexcept;
// FP16 projection rounding precedes FP16 residual addition, matching the
// independent projection and acc operators. Weights retain their GGUF layout.
void matvec_m1_residual(ggml_type type,const void* packed,std::uint64_t row_bytes,
    std::int64_t rows,std::int64_t columns,const void* input,bool aligned_input,
    const void* residual,void* output,cudaStream_t stream);
}
