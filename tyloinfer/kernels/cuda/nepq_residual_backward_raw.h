#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cstdint>

namespace mfq_jit_nepq_residual {
/** Accumulate the original sparse residual gradient in caller-owned output. */
void backward_input(const __half* dictionary, const std::int16_t* first,
    const std::int16_t* second, const __half* output_gradient, __half* input_gradient,
    std::int64_t input_rows, int rows, int width, std::int64_t blocks_per_row, int position_bits,
    int block_vectors, cudaStream_t stream);

/** Accumulate transposed residual records and optionally apply the rotation. */
void backward_transpose(const __half* dictionary, const std::int32_t* transpose_offsets,
    const std::int32_t* transpose_rows, const std::int16_t* transpose_dictionary,
    const __half* output_gradient, const std::int8_t* rotation_signs, __half* input_gradient,
    std::int64_t input_rows, int rows, int width, std::int64_t vectors, bool fuse_rotation,
    cudaStream_t stream);
}
