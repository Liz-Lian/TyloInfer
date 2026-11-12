#pragma once
#include "numeric_type.h"
#include <cuda_runtime_api.h>
#include <cstdint>

namespace tyloi::cuda::sampling_raw {
/** Select argmax, optionally using caller-owned chunk buffers. */
void greedy(const void*, int64_t*, float*, int64_t*, int rows, int vocab, NumericType, cudaStream_t);
/** Sample temperature-scaled full softmax. */
void softmax(const void*, const float*, int64_t*, int rows, int vocab, float temperature, NumericType, cudaStream_t);
/** Select packed top-k rows, with an optional device replay condition. */
void compact(const void*, int64_t*, int64_t*, int64_t*, int rows, int vocab, int top_k,
    const volatile int32_t* enabled, NumericType, cudaStream_t);
/** Execute one of the original top-k/top-p sampling strategies. */
void topk(const void*, const float*, int64_t*, void*, void*, void*, void*, int rows, int vocab,
    int top_k, float temperature, float top_p, bool radix, bool keys, int items,
    bool bitonic, NumericType, cudaStream_t);
/** Increment a device token-count vector in place. */
void counts(int32_t*, const int64_t*, int64_t count, int vocab, cudaStream_t);
/** Apply penalties to logits in place. */
void penalties(void*, const int32_t*, int vocab, float presence, float frequency,
    float repetition, NumericType, cudaStream_t);
}
