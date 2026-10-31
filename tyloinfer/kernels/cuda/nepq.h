#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace mfq::cuda::hadamard {
// Contiguous rows, normalized Sylvester transform in FP32, preserving dtype. Forward is
// H(Dx), adjoint is D(Hx); null signs mean identity. Input/output may alias.
void apply_f16(const void* input, const std::int8_t* signs, void* output,
    std::int64_t rows, std::int64_t width, int block_size, bool adjoint,
    cudaStream_t stream);
void apply_f32(const float* input, const std::int8_t* signs, float* output,
    std::int64_t rows, std::int64_t width, int block_size, bool adjoint,
    cudaStream_t stream);
}
