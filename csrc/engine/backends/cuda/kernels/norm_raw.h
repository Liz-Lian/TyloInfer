#pragma once

#include "numeric_type.h"
#include <cuda_runtime_api.h>
#include <cstdint>

namespace tyloi::cuda::norm_raw {

/** Normalize contiguous F32 or F16 rows. */
void rms(const void*, const float*, void*, int rows, int width, float eps, float offset,
    NumericType, cudaStream_t);
/** Normalize contiguous F32 rows by their L2 norm. */
void l2(const float*, float*, int rows, int width, float eps, cudaStream_t);
/** Normalize GDN output and apply a row-strided gate. */
void gdn_gate(const float*, const void*, const float*, void*, int rows, int heads, int tokens,
    int width, double eps, bool silu, bool double_sigmoid, bool warp_tail,
    std::int64_t batch_stride, std::int64_t token_stride, NumericType gate_type,
    NumericType output_type, cudaStream_t);
/** Normalize groups after applying a SiLU gate. */
void grouped_gated(const void*, const void*, const float*, void*, std::int64_t blocks,
    int group_size, int groups, float eps, NumericType input_type, NumericType gate_type, cudaStream_t);
/** Normalize independent groups. */
void grouped(const void*, const float*, void*, int rows, int width, int weight_rows,
    float eps, float offset, NumericType, cudaStream_t);
/** Apply Qwen BF16 RMS normalization. */
void qwen_bf16(const void*, const float*, void*, int rows, int width, float eps, float offset, cudaStream_t);
/** Compute input squares for paired Qwen RMS normalization. */
void qwen_pair_square(const void*, const void*, float*, std::int64_t first_count,
    std::int64_t total_count, int blocks, cudaStream_t);
/** Finalize paired Qwen RMS normalization from row means. */
void qwen_pair_finish(const void*, const void*, const float*, const float*, const float*, void*, void*,
    std::int64_t first_count, std::int64_t total_count, int width, int blocks, float eps, float offset, cudaStream_t);
/** Apply MiniCPM QK normalization, RoPE and in-place cache writes. */
void minicpm_qk(const void*, const void*, const void*, const float*, const float*,
    const std::int64_t*, const std::int64_t*, const float*, const float*, void*, void*, void*,
    int batch, int table_len, int cache_len, float eps, float offset, cudaStream_t);
/** Normalize two FP16 inputs into F32 outputs. */
void rms_pair_f16_f32(const void*, const void*, const float*, const float*, float*, float*,
    int first_rows, int second_rows, int width, float eps, float offset, cudaStream_t);

} // namespace tyloi::cuda::norm_raw
