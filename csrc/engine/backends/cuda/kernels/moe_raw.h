#pragma once

#include "numeric_type.h"
#include <cuda_runtime_api.h>
#include <cstdint>

namespace tyloi::cuda::moe_raw {
/** Select the original router kernel after host-side policy resolution. */
enum class RouterKernel { generic, cached, fixed_128, fixed_256, fixed_sqrt_256, fixed_512 };
/** Describe original top-k routing inputs, flags and output buffers. */
struct Router {
    const void* logits;
    const float* bias;
    int32_t* ids;
    float* weights;
    int rows, experts, bias_stride, top_k;
    bool sigmoid, sqrt_softplus, normalize, delayed_softmax;
    float norm_floor, scale;
    NumericType type;
    RouterKernel kernel;
};
/** Launch the original router kernel on the supplied stream. */
void topk(const Router&, cudaStream_t);
/** Compute the original sqrt-softplus weights for selected experts. */
void sqrt_weights(const void* logits, const int32_t* ids, float* weights,
    int rows, int experts, int top_k, float floor, float scale, NumericType, cudaStream_t);
/** Describe caller-owned route-map workspaces and optional additional tilings. */
struct RouteMap {
    const int32_t* ids;
    int32_t *counts, *cursors, *ids_dst, *expert_bounds, *tile_bounds, *tile_experts;
    int32_t *secondary_bounds, *secondary_experts, *tertiary_bounds, *tertiary_experts;
    int pairs, experts, tile, secondary_tile, tertiary_tile;
};
/** Build the original count, prefix, tile and scatter maps on one stream. */
void build_map(const RouteMap&, cudaStream_t);
/** Reduce route-weighted outputs in their original storage precision. */
void reduce(const void*, const float*, void*, int tokens, int routes, int width, NumericType, cudaStream_t);
/** Apply the original split SiLU or GELU gate. */
void glu(const void*, void*, int rows, int width, bool gelu, cudaStream_t);
/** Fill the original single- and double-precision sigmoid tables. */
void sigmoid_table(void*, cudaStream_t);
/** Apply the rounded FP16 SwiGLU path using the selected table half. */
void rounded(const void*, const void*, const void*, void*, int64_t count, bool double_table, cudaStream_t);
/** Multiply routing weights in place by selected expert scales. */
void expert_scale(float*, const int32_t*, const float*, int count, cudaStream_t);
/** Add a sigmoid-gated shared output. */
void add_shared(const void*, const void*, const float*, void*, int rows, int width, cudaStream_t);
/** Reduce routed outputs and add a sigmoid-gated shared output. */
void reduce_shared(const void*, const float*, const void*, const float*, void*, int tokens, int routes, int width, cudaStream_t);
#if TYLOINFER_ENABLE_MFQ_COMPAT
/** Quantize a shared expert input using the caller's descriptors. */
void quantize_shared(const void*, const int64_t*, int rows, int width, int groups, cudaStream_t);
/** Build routed NINT row metadata and optionally quantize shared input. */
void quantize_nint_rows(const void*, const int64_t*, int rows, int width, int groups, bool quantize,
    int64_t*, float*, float*, void*, const int32_t*, const int32_t*, int pairs,
    int experts, int output_rows, int local_experts, int blocks, cudaStream_t);
/** Quantize one activation matrix into caller-owned input workspaces. */
void quantize(const void*, int8_t*, float*, int rows, int width, int padded_width,
    int groups, int group_size, cudaStream_t);
/** Describe packed NINT8-0 inputs, route maps, and caller-owned output. */
struct Nint8 {
    const uint8_t* q;
    const void* scale;
    const void* input;
    const int8_t* qx;
    const float* xscale;
    const int32_t *ids, *local, *ids_dst, *expert_bounds, *tile_bounds, *tile_experts;
    void* output;
    int tokens, routes, experts, output_rows, groups, width;
    bool routed_input;
};
/** Run one of the original NINT8-0 direct token kernels. */
void nint8_direct(const Nint8&, int items, cudaStream_t);
/** Run the original persistent grouped NINT8-0 kernel. */
void nint8_persistent(const Nint8&, int blocks, cudaStream_t);
/** Run the original grouped NINT8-0 tile kernel. */
void nint8_grouped(const Nint8&, int blocks, int max_tiles, cudaStream_t);
/** Run the original routed NINT8-0 MMA specialization. */
void nint8_mma(const Nint8&, int blocks, int rows_per_tile, bool coarse, cudaStream_t);
#endif
}
