#pragma once
#include "numeric_type.h"
#include <cuda_runtime_api.h>
#include <cstdint>

namespace tyloi::cuda::attention_raw {
/** Describe attention storage, geometry, and caller-owned split workspace. */
struct Args {
    const void* q = nullptr;
    const void* k = nullptr;
    const void* v = nullptr;
    const int64_t* lengths = nullptr;
    void* out = nullptr;
    float* partial_o = nullptr;
    float* partial_m = nullptr;
    float* partial_l = nullptr;
    int batch = 0, heads = 0, kv_heads = 0, tokens = 0, keys = 0, width = 0;
    int parts = 1, workspace_parts = 1, window = 0, maximum_cache_position = -1;
    float scale = 0;
    bool causal = false, dynamic_parts = false;
    NumericType type = NumericType::f32;
};
/** Run the original fixed-width FP16 flash prefill kernel. */
void flash256(const Args&, cudaStream_t);
/** Run full or windowed attention using optional split workspace. */
void full(const Args&, cudaStream_t);
/** Run unsplit cache decode attention. */
void decode(const Args&, cudaStream_t);
/** Run split cache decode with supplied workspace. */
void decode_split(const Args&, cudaStream_t);
/** Run circular-cache sliding-window attention using optional workspace. */
void cache_swa(const Args&, cudaStream_t);
}
