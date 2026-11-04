#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

std::vector<::tyloi::DenseTensor> moe_topk_cuda(
    ::tyloi::DenseTensor logits, int64_t top_k, bool use_sigmoid, bool use_sqrt_softplus,
    bool normalize,
    bool delayed_softmax, MfqOptional<::tyloi::DenseTensor> bias, double norm_floor,
    double scale);
::tyloi::DenseTensor moe_sqrtsoftplus_weights_cuda(
    ::tyloi::DenseTensor logits, ::tyloi::DenseTensor ids, double norm_floor, double scale);
std::vector<::tyloi::DenseTensor> moe_build_expert_map_cuda(
    ::tyloi::DenseTensor ids, int64_t n_experts, int64_t tile_m);
std::vector<::tyloi::DenseTensor> moe_build_expert_maps_cuda(
    ::tyloi::DenseTensor ids, int64_t n_experts, int64_t tile_m,
    int64_t secondary_tile_m, int64_t tertiary_tile_m);
void moe_quantize_shared_input_cuda(::tyloi::DenseTensor input,
    ::tyloi::DenseTensor descriptors,int64_t total_groups);
void moe_quantize_shared_nint_rows_cuda(::tyloi::DenseTensor input,
    ::tyloi::DenseTensor descriptors,int64_t total_groups,bool quantize,
    ::tyloi::DenseTensor offsets,::tyloi::DenseTensor scales,
    ::tyloi::DenseTensor minima,::tyloi::DenseTensor row_metadata,
    ::tyloi::DenseTensor expert_local,::tyloi::DenseTensor ids,
    int64_t output_rows,int64_t local_experts);
::tyloi::DenseTensor mfe_nint_matmul_ws_cuda(
    ::tyloi::DenseTensor q_packed, ::tyloi::DenseTensor row_q_bits,
    ::tyloi::DenseTensor row_q_bit_offsets,
    ::tyloi::DenseTensor sub_scale, ::tyloi::DenseTensor sub_min,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor neuron_min,
    ::tyloi::DenseTensor x, ::tyloi::DenseTensor ids,
    ::tyloi::DenseTensor expert_local, int64_t n_experts,
    int64_t n_local_experts, int64_t out_per_expert, int64_t gs,
    int64_t epilogue_mode, bool route_map_ready, bool input_quantized,
    ::tyloi::DenseTensor out, ::tyloi::DenseTensor qx,
    ::tyloi::DenseTensor xscale,
    ::tyloi::DenseTensor ids_dst,
    ::tyloi::DenseTensor expert_bounds,
    ::tyloi::DenseTensor tile_bounds,
    ::tyloi::DenseTensor tile_experts,
    int64_t route_tile_m,
    int64_t pool_phase);
::tyloi::DenseTensor nint8_zero_moe_grouped_matmul_pool_ws_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor scale, ::tyloi::DenseTensor x,
    ::tyloi::DenseTensor ids, ::tyloi::DenseTensor expert_local, int64_t n_experts,
    int64_t n_local_experts, int64_t out_per_expert, bool route_map_ready,
    bool input_quantized, bool use_f16_mma, ::tyloi::DenseTensor out, ::tyloi::DenseTensor qx,
    ::tyloi::DenseTensor xscale, ::tyloi::DenseTensor counts, ::tyloi::DenseTensor cursors,
    ::tyloi::DenseTensor ids_dst, ::tyloi::DenseTensor expert_bounds,
    ::tyloi::DenseTensor tile_bounds, ::tyloi::DenseTensor tile_experts,
    int64_t route_tile_m);
::tyloi::DenseTensor nvq_moe_grouped_matmul_pool_ws_cuda(
    ::tyloi::DenseTensor indices, ::tyloi::DenseTensor aux, ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor codebook, ::tyloi::DenseTensor x,
    ::tyloi::DenseTensor ids, ::tyloi::DenseTensor expert_local, int64_t n_experts,
    int64_t pool_experts, int64_t out_per_expert, int64_t neuron_len,
    int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    bool input_quantized, ::tyloi::DenseTensor out, ::tyloi::DenseTensor qx,
    ::tyloi::DenseTensor xscale, ::tyloi::DenseTensor ids_dst, ::tyloi::DenseTensor expert_bounds,
    ::tyloi::DenseTensor tile_bounds, ::tyloi::DenseTensor tile_experts);
::tyloi::DenseTensor nvq_moe_grouped_matmul_pool_f16_cuda(
    ::tyloi::DenseTensor indices, ::tyloi::DenseTensor aux, ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor codebook, ::tyloi::DenseTensor x,
    ::tyloi::DenseTensor expert_local, int64_t n_experts, int64_t pool_experts,
    int64_t out_per_expert, int64_t neuron_len, int64_t gs,
    int64_t sub_bits, int64_t format, int64_t sign_mode,
    ::tyloi::DenseTensor out, ::tyloi::DenseTensor ids_dst, ::tyloi::DenseTensor expert_bounds,
    ::tyloi::DenseTensor tile_bounds, ::tyloi::DenseTensor tile_experts);
::tyloi::DenseTensor nvq_moe_grouped_matmul_hetero_f16_cuda(
    ::tyloi::DenseTensor weight_ptrs, ::tyloi::DenseTensor weight_sizes,
    ::tyloi::DenseTensor pool_params, ::tyloi::DenseTensor expert_pool,
    ::tyloi::DenseTensor expert_local, ::tyloi::DenseTensor x,
    int64_t n_experts, int64_t out_per_expert, int64_t neuron_len,
    int64_t route_tile_m, ::tyloi::DenseTensor out,
    ::tyloi::DenseTensor ids_dst,
    ::tyloi::DenseTensor expert_bounds, ::tyloi::DenseTensor tile_bounds,
    ::tyloi::DenseTensor tile_experts, int64_t format_group,
    bool masked_experts);
::tyloi::DenseTensor nvq_moe_grouped_matmul_hetero_ws_cuda(
    ::tyloi::DenseTensor weight_ptrs, ::tyloi::DenseTensor weight_sizes,
    ::tyloi::DenseTensor pool_params, ::tyloi::DenseTensor expert_pool,
    ::tyloi::DenseTensor expert_local, ::tyloi::DenseTensor x,
    ::tyloi::DenseTensor ids, int64_t n_experts, int64_t out_per_expert,
    int64_t neuron_len, bool input_quantized, ::tyloi::DenseTensor out,
    ::tyloi::DenseTensor qx, ::tyloi::DenseTensor xscale);
::tyloi::DenseTensor nvq_moe_grouped_matmul_active_ws_cuda(
    ::tyloi::DenseTensor weight_ptrs, ::tyloi::DenseTensor weight_sizes,
    ::tyloi::DenseTensor pool_params, ::tyloi::DenseTensor expert_pool,
    ::tyloi::DenseTensor expert_local, ::tyloi::DenseTensor x,
    ::tyloi::DenseTensor ids, int64_t n_experts, int64_t out_per_expert,
    int64_t neuron_len, bool input_quantized, ::tyloi::DenseTensor out,
    ::tyloi::DenseTensor qx, ::tyloi::DenseTensor xscale,
    ::tyloi::DenseTensor active_plan);
::tyloi::DenseTensor nepq_moe_grouped_matmul_pool_ws_cuda(
    ::tyloi::DenseTensor indices, ::tyloi::DenseTensor aux, ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor table_pool, ::tyloi::DenseTensor bank_ids,
    ::tyloi::DenseTensor grouped_table_pool, ::tyloi::DenseTensor x, ::tyloi::DenseTensor ids,
    ::tyloi::DenseTensor expert_local, int64_t n_experts, int64_t pool_experts,
    int64_t out_per_expert, int64_t neuron_len, int64_t sub_bits, int64_t format,
    bool input_quantized, ::tyloi::DenseTensor out, ::tyloi::DenseTensor qx,
    ::tyloi::DenseTensor xscale, ::tyloi::DenseTensor ids_dst, ::tyloi::DenseTensor expert_bounds,
    ::tyloi::DenseTensor tile_bounds, ::tyloi::DenseTensor tile_experts);
::tyloi::DenseTensor nepq_moe_grouped_matmul_pool_f16_cuda(
    ::tyloi::DenseTensor indices, ::tyloi::DenseTensor aux, ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor table_pool,
    ::tyloi::DenseTensor bank_ids, ::tyloi::DenseTensor x, ::tyloi::DenseTensor expert_local,
    int64_t n_experts, int64_t pool_experts, int64_t out_per_expert,
    int64_t neuron_len, int64_t sub_bits, int64_t format,
    ::tyloi::DenseTensor out, ::tyloi::DenseTensor ids_dst, ::tyloi::DenseTensor expert_bounds,
    ::tyloi::DenseTensor tile_bounds, ::tyloi::DenseTensor tile_experts);
::tyloi::DenseTensor nepq_sparse_residual_grouped_cuda(
    ::tyloi::DenseTensor dictionary, ::tyloi::DenseTensor first, ::tyloi::DenseTensor second,
    ::tyloi::DenseTensor input, ::tyloi::DenseTensor route_ids, ::tyloi::DenseTensor expert_local,
    int64_t out_per_expert, int64_t position_bits, int64_t block_vectors,
    ::tyloi::DenseTensor output);
::tyloi::DenseTensor moe_weighted_reduce_cuda(::tyloi::DenseTensor pair_output, ::tyloi::DenseTensor weights);
::tyloi::DenseTensor moe_swiglu_split_cuda(::tyloi::DenseTensor gate_up);
::tyloi::DenseTensor moe_swiglu_rounded_cuda(
    ::tyloi::DenseTensor gate, ::tyloi::DenseTensor up,
    ::tyloi::DenseTensor sigmoid_table);
::tyloi::DenseTensor moe_swiglu_sigmoid_table_cuda();
::tyloi::DenseTensor moe_geglu_split_cuda(::tyloi::DenseTensor gate_up);
::tyloi::DenseTensor moe_apply_expert_scale_cuda(
    ::tyloi::DenseTensor weights, ::tyloi::DenseTensor ids, ::tyloi::DenseTensor scales);
::tyloi::DenseTensor moe_add_shared_gate_cuda(
    ::tyloi::DenseTensor routed, ::tyloi::DenseTensor shared, ::tyloi::DenseTensor gate_logits);
::tyloi::DenseTensor moe_weighted_reduce_shared_gate_cuda(
    ::tyloi::DenseTensor pair_output, ::tyloi::DenseTensor weights,
    ::tyloi::DenseTensor shared, ::tyloi::DenseTensor gate_logits);
::tyloi::DenseTensor mxfp4_moe_grouped_matmul_pool_f16_cuda(
    ::tyloi::DenseTensor values, ::tyloi::DenseTensor scales, ::tyloi::DenseTensor input,
    ::tyloi::DenseTensor ids, ::tyloi::DenseTensor expert_local,
    int64_t global_experts, int64_t pool_experts,
    int64_t out_per_expert, int64_t neuron_len,
    ::tyloi::DenseTensor output, ::tyloi::DenseTensor ids_dst,
    ::tyloi::DenseTensor expert_bounds, ::tyloi::DenseTensor tile_bounds,
    ::tyloi::DenseTensor tile_experts);
