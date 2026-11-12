#pragma once

#include "numeric_type.h"
#include <cuda_runtime_api.h>
#include <cstdint>

namespace tyloi::cuda::acc_raw {

/** Add two contiguous activation buffers. */
void add(const void*, const void*, void*, int count, NumericType, cudaStream_t);
/** Add and normalize contiguous rows, with the requested normalized output type. */
void rms(const void*, const void*, const float*, void*, void*, int rows, int width,
    float eps, float offset, NumericType input_type, NumericType output_type, cudaStream_t);
/** Compute Gemma attention residual and three pre-normalized outputs. */
void gemma_pre(const void*, const void*, const float*, const float*, const float*, const float*,
    void*, void*, float*, void*, int rows, int width, float eps, cudaStream_t);
/** Merge Gemma dense, expert and residual branches. */
void gemma_merge(const void*, const void*, const void*, const float*, const float*, const float*,
    const void*, void*, int rows, int width, float eps, cudaStream_t);
/** Commit one graph decode token and advance counters in place. */
void commit(const std::int64_t*, std::int64_t*, std::int64_t*, std::int64_t*,
    std::int64_t*, std::int64_t*, cudaStream_t);

} // namespace tyloi::cuda::acc_raw
