#pragma once
#include "numeric_type.h"
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace tyloi::cuda::embedding_raw {
/** Look up dense rows using the original aligned-vector or scalar launch. */
void dense(const void*, const int64_t*, void*, int tokens, int width, int vocabulary,
    NumericType, cudaStream_t);
#if TYLOINFER_ENABLE_MFQ_COMPAT
/** Decode canonical NINT embedding rows. */
void nint(const uint8_t*, const uint8_t*, const int64_t*, const uint8_t*, const uint8_t*,
    const float*, const float*, const int64_t*, void*, int tokens, int vocabulary,
    int groups, int group_size, int width, cudaStream_t);
/** Decode selected packed rows, optionally permuted by inverse indices. */
void selected(const uint8_t*, const int32_t*, const int64_t*, void*, std::size_t count,
    int width, cudaStream_t);
/** Decode NINT8 zero-point rows. */
void nint8(const uint8_t*, const void*, const int64_t*, void*, int tokens,
    int vocabulary, int groups, int width, cudaStream_t);
#endif
}
