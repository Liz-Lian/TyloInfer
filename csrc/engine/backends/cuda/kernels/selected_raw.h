#pragma once

#include "sparse_mma_raw.h"

#include <cuda_runtime_api.h>
#include <cstdint>

namespace tyloi::cuda::selected_raw {

/** Select the original Torch or native sigmoid arithmetic. */
enum class Sigmoid { torch_float, native_float, native_double };

/** Describe four tensor strides in elements. */
struct Layout { int64_t batch, token, head, column; };

/** Describe the optional fused gate and its output storage. */
struct Epilogue {
    const void* gate = nullptr;
    void* output = nullptr;
    bool float_gate = false;
    bool half_output = false;
    Sigmoid sigmoid = Sigmoid::native_float;
    Layout layout{};
};

/** Pad and validate selected indices for the MMA kernel. */
void prepare_indices(const int32_t* source, int32_t* indices, void* mask,
    int64_t rows, int source_count, int padded_count, int cache_length, cudaStream_t stream);

/** Run the original generic selected or causal attention kernel. */
void sparse(const void* query, const void* key, const void* value, const int32_t* indices,
    float* scratch_or_output, int64_t batch, int64_t heads, int64_t kv_heads,
    int64_t tokens, int64_t width, int64_t cache_length, int64_t selected,
    float scale, bool half_query, const void* positions, bool positions_i64,
    Epilogue epilogue, sparse_mma_raw::Hot hot, cudaStream_t stream);

/** Apply the attention gate to strided floating-point inputs. */
void gated(const void* attended, bool half_input, Layout layout, Epilogue epilogue,
    int64_t count, int64_t heads, int64_t tokens, int64_t width, cudaStream_t stream);

/** Expand block selections into token indices. */
void block_indices(const void* blocks, bool blocks_i64, const void* positions,
    bool positions_i64, int32_t* output, int64_t rows, int tokens,
    int block_count, int pool, int budget, int columns, cudaStream_t stream);

/** Prepare the contiguous query and masked causal indices for MMA. */
void prepare_causal(const void* query, bool half_query, Layout layout, float* compact_query,
    int64_t query_count, int64_t heads, int64_t tokens, int64_t width,
    const void* positions, bool positions_i64, int32_t* indices, void* mask,
    int64_t index_count, int64_t columns, int64_t padded_columns,
    int64_t capacity, cudaStream_t stream);

/** Combine stream-K partials and apply the original gated epilogue. */
void reduce_gated(const float* partial, const void* metadata, const void* gate,
    void* output, Layout gate_layout, int64_t rows, int tokens, int heads, int kv_heads,
    int width, int total_blocks, int parts, int query_columns, int head_columns,
    bool float_gate, bool half_output, Sigmoid sigmoid, bool parallel, cudaStream_t stream);

} // namespace tyloi::cuda::selected_raw
