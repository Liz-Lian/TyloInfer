#pragma once

#include "numeric_type.h"
#include <cuda_runtime_api.h>
#include <cstdint>

namespace tyloi::cuda::kv_cache_raw {
/** Write selected positions in place, preserving cache storage. */
void write(void* keys, void* values, const void* input_keys, const void* input_values,
    const int64_t* positions, int batch, int heads, int tokens, int width,
    int capacity, int positions_rank, NumericType type, cudaStream_t stream);
/** Append the newest tokens in place to a circular cache. */
void ring(void* keys, void* values, const void* input_keys, const void* input_values,
    int64_t start, int batch, int heads, int tokens, int width, int capacity,
    NumericType type, cudaStream_t stream);
/** Write the newest explicitly positioned tokens to a circular cache. */
void ring_positions(void* keys, void* values, const void* input_keys, const void* input_values,
    const int64_t* positions, int batch, int heads, int tokens, int width, int capacity,
    NumericType type, cudaStream_t stream);
}
