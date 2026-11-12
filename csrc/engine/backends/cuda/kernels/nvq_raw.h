#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace tyloi::cuda::nvq_raw {
/** Describe packed weights and their original CUDA decoding geometry. */
struct Weight {
    const std::uint8_t *indices, *aux, *state;
    std::int64_t indices_bytes, aux_bytes, state_bytes;
    const float* scale;
    const std::int8_t* table;
    std::int64_t table_bytes;
    const std::uint8_t* banks;
    int rows, width, group_size, state_bits, format, sign_mode, table_stride;
};
/** Describe caller-owned activation and output workspaces for a projection. */
struct Projection {
    Weight weight;
    const void* input;
    std::int8_t* quantized;
    float* scales;
    void* output;
    const void* residual;
    const void* gate;
    int rows, warps, gate_mode;
};
/** Describe caller-owned NEPQ route maps and expert outputs. */
struct Grouped {
    Projection projection;
    const std::int8_t* grouped_table;
    const std::int32_t *ids, *ids_dst, *expert_bounds, *tile_bounds, *tile_experts, *expert_local;
    int tokens, routes, experts, pool_experts, output_rows, grouped_stride;
    bool routed_input;
};
/** Quantize a projection's activation using the original gs24 path. */
void quantize(const Projection&, cudaStream_t);
/** Return the required NEPQ grouped table stride. */
int nepq_grouped_stride(int format, int stride);
/** Run the original online-F16 NVQ pooled routed projection. */
void nvq_grouped_f16(const Grouped&, cudaStream_t);
/** Run the original online-F16 NEPQ pooled routed projection. */
void nepq_grouped_f16(const Grouped&, cudaStream_t);
/** Report whether the format has the original dense-group layout. */
bool dense_format(int format);
/** Dispatch the original NVQ pooled route; return whether its dense path ran. */
bool nvq_grouped(const Grouped&, cudaStream_t);
/** Run an already-quantized NEPQ routed projection. */
void nepq_grouped(const Grouped&, cudaStream_t);
/** Describe heterogeneous pooled weights, routes, and caller-owned buffers. */
struct Hetero {
    const std::int64_t *weight_ptrs, *weight_sizes;
    const std::int32_t *pool_params, *expert_pool, *expert_local;
    const void* input;
    void* output;
    const std::int32_t *ids_dst, *expert_bounds, *tile_bounds, *tile_experts;
    int tokens, routes, experts, pools, output_rows, width, tile_m, format_group;
    bool masked_experts, routed_input;
};
/** Describe prequantized activation and optional active-route plan. */
struct HeteroQuant {
    Hetero route;
    const std::int8_t* codes;
    const float* scales;
    const std::int32_t *ids, *active;
};
/** Dispatch heterogeneous quantized routes. */
void hetero_quant(const HeteroQuant&, cudaStream_t);
/** Return whether the original heterogeneous format group is supported. */
bool hetero_format(std::int64_t format_group);
/** Dispatch heterogeneous F16 routes; false reports a shared-memory opt-in failure. */
bool hetero_f16(const Hetero&, cudaStream_t);
/** Apply the original shared-bank NEPQ GEMV. */
void nepq_gemv(const Projection&, cudaStream_t);
/** Apply the original shared-bank NEPQ integer-MMA projection. */
void nepq_mmq(const Projection&, cudaStream_t);
/** Apply the original shared-bank NEPQ online F16 projection. */
void nepq_gemm(const Projection&, cudaStream_t);
/** Describe two packed projections sharing one activation workspace. */
struct Pair { Projection first; Weight second; };
/** Describe the original fused gate/up output quantization buffers. */
struct Ffn { Pair pair; std::int8_t* output_codes; float *output_scales, *scratch; int down_group; };
/** Test for the original NVQ2 format. */
bool nvq2_format(std::int64_t format);
/** Apply the original fused SwiGLU projection, optionally the ordered NVQ2 variant. */
void swiglu(const Pair&, bool ordered, cudaStream_t);
/** Apply the original fused SwiGLU and quantize its output. */
void ffn(const Ffn&, cudaStream_t);
/** Quantize once and apply both original NVQ projections. */
void multi2(const Pair&, cudaStream_t);
/** Apply the original quantized GEMV dispatch and optional activation quantization. */
void gemv(const Projection&, bool quantize, bool gated, int method, cudaStream_t);
/** Apply the original quantized integer-MMA projection. */
void mmq(const Projection&, bool gated, cudaStream_t);
/** Return whether a format supports the original fused residual path. */
bool group_exec(int format);
/** Apply the original aligned-group residual GEMV. */
void gemv_residual(const Projection&, cudaStream_t);
/** Run the original F16 online-dequant projection. */
void gemm(const Weight&, const void* input, void* output, int rows, cudaStream_t);
/** Decode NVQ or NPQ weights to an F16 matrix on the supplied stream. */
void decode(const Weight&, void* output, cudaStream_t);
/** Decode shared-bank NEPQ weights to an F16 matrix on the supplied stream. */
void decode_nepq(const Weight&, void* output, cudaStream_t);
/** Decode selected NVQ embedding rows to F16 on the supplied stream. */
void embedding(const Weight&, const std::int64_t* ids, int tokens, void* output, cudaStream_t);
}
