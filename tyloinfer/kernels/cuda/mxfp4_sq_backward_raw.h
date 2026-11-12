#pragma once

#include "mxfp4_sq_raw.h"
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cstdint>

namespace mfq_jit_sq4 {
/** Launch the original small-row MXFP4-SQ partial input gradient. */
void backward_matrix(const std::uint8_t* blob, const std::uint8_t* row_q,
    const std::int32_t* row_symbol_byte_offsets, const std::int32_t* row_auxiliary,
    const __half* output_gradient, float* partials, tyloi::sq::Layout layout,
    int rows, int output_tile, cudaStream_t stream);
}
