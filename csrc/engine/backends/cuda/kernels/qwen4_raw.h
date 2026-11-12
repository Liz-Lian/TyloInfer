#pragma once

#include "numeric_type.h"
#include <cuda_runtime_api.h>
#include <cstdint>

namespace tyloi::cuda::qwen4_raw {
/** Pool live index rows and produce adjusted positions. */
void live_pool(const void*, const void*, const std::int64_t*, void*, std::int64_t*, void*,
    int, int, int, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    NumericType, bool, cudaStream_t);
/** Fill decode cache and position buffers. */
void fill_positions(std::int64_t*, void*, std::int64_t, std::int64_t, std::int64_t,
    std::int64_t, bool, cudaStream_t);
/** Apply gated residual bottleneck or injection activation. */
void activation(const void*, void*, std::int64_t, float, NumericType, bool, cudaStream_t);
/** Mix normalized residual streams. */
void mix(const void*, const void*, void*, std::int64_t, int, int, NumericType, NumericType, cudaStream_t);
/** Apply the residual update. */
void post(const void*, const void*, const void*, void*, std::int64_t, int, int, NumericType, cudaStream_t);
/** Apply the residual update and grouped RMS normalization. */
void post_norm(const void*, const void*, const void*, const float*, void*, void*,
    std::int64_t, int, int, float, NumericType, NumericType, NumericType, cudaStream_t);
/** Compute live pooled-block scores. */
void live_scores(const void*, const void*, const std::int64_t*, float*, int, int, int,
    int, int, std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    NumericType, NumericType, cudaStream_t);
/** Reduce one stage of the live top-k selection. */
cudaError_t prepare_live_topk();
/** Launch one stage of live top-k reduction. */
void live_topk(const float*, const std::int64_t*, float*, std::int64_t*, const std::int64_t*,
    int, int, int, int, int, int, bool, cudaStream_t);
}
