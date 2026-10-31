#pragma once
#include "ggml.h"
#include <cuda_runtime_api.h>
#include <cstdint>

namespace mfq::cuda::gguf {
// F32 GR bottleneck SiLU + native Q8_1 packing and injection gate. The
// activation rounding and Q8 packet match the separate operations exactly.
void prepare_residual_activation(const float* down,const float* injection,
    std::int64_t tokens,std::int64_t columns,std::int64_t streams,
    void* packed,float* injection_output,cudaStream_t stream);
bool residual_mix_supported(ggml_type type) noexcept;
// Original compressed weights and native Q8_1 activations. FP32 projection,
// sigmoid, product rounding and stream reduction match separate operations.
void residual_mix(ggml_type type,const void* packed,std::uint64_t stride,
    std::int64_t rows,std::int64_t columns,const void* input,std::int64_t tokens,
    const float* normalized,std::int64_t streams,float* output,cudaStream_t stream);
}
