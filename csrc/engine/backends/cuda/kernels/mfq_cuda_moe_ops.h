#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

std::vector<::tyloi::Tensor> moe_topk_cuda(
    ::tyloi::Tensor logits, int64_t top_k, bool use_sigmoid, bool use_sqrt_softplus,
    bool normalize,
    bool delayed_softmax, MfqOptional<::tyloi::Tensor> bias, double norm_floor,
    double scale);
::tyloi::Tensor moe_sqrtsoftplus_weights_cuda(
    ::tyloi::Tensor logits, ::tyloi::Tensor ids, double norm_floor, double scale);
std::vector<::tyloi::Tensor> moe_build_expert_map_cuda(
    ::tyloi::Tensor ids, int64_t n_experts, int64_t tile_m);
std::vector<::tyloi::Tensor> moe_build_expert_maps_cuda(
    ::tyloi::Tensor ids, int64_t n_experts, int64_t tile_m,
    int64_t secondary_tile_m, int64_t tertiary_tile_m);
void moe_quantize_shared_input_cuda(::tyloi::Tensor input,
    ::tyloi::Tensor descriptors,int64_t total_groups);
void moe_quantize_shared_nint_rows_cuda(::tyloi::Tensor input,
    ::tyloi::Tensor descriptors,int64_t total_groups,bool quantize,
    ::tyloi::Tensor offsets,::tyloi::Tensor scales,
    ::tyloi::Tensor minima,::tyloi::Tensor row_metadata,
    ::tyloi::Tensor expert_local,::tyloi::Tensor ids,
    int64_t output_rows,int64_t local_experts);
::tyloi::Tensor mfe_nint_matmul_ws_cuda(
    ::tyloi::Tensor q_packed, ::tyloi::Tensor row_q_bits,
    ::tyloi::Tensor row_q_bit_offsets,
    ::tyloi::Tensor sub_scale, ::tyloi::Tensor sub_min,
    ::tyloi::Tensor neuron_scale, ::tyloi::Tensor neuron_min,
    ::tyloi::Tensor x, ::tyloi::Tensor ids,
    ::tyloi::Tensor expert_local, int64_t n_experts,
    int64_t n_local_experts, int64_t out_per_expert, int64_t gs,
    int64_t epilogue_mode, bool route_map_ready, bool input_quantized,
    ::tyloi::Tensor out, ::tyloi::Tensor qx,
    ::tyloi::Tensor xscale,
    ::tyloi::Tensor ids_dst,
    ::tyloi::Tensor expert_bounds,
    ::tyloi::Tensor tile_bounds,
    ::tyloi::Tensor tile_experts,
    int64_t route_tile_m,
    int64_t pool_phase);
::tyloi::Tensor nint8_zero_moe_grouped_matmul_pool_ws_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor scale, ::tyloi::Tensor x,
    ::tyloi::Tensor ids, ::tyloi::Tensor expert_local, int64_t n_experts,
    int64_t n_local_experts, int64_t out_per_expert, bool route_map_ready,
    bool input_quantized, bool use_f16_mma, ::tyloi::Tensor out, ::tyloi::Tensor qx,
    ::tyloi::Tensor xscale, ::tyloi::Tensor counts, ::tyloi::Tensor cursors,
    ::tyloi::Tensor ids_dst, ::tyloi::Tensor expert_bounds,
    ::tyloi::Tensor tile_bounds, ::tyloi::Tensor tile_experts,
    int64_t route_tile_m);
::tyloi::Tensor nvq_moe_grouped_matmul_pool_ws_cuda(
    ::tyloi::Tensor indices, ::tyloi::Tensor aux, ::tyloi::Tensor sub_scale,
    ::tyloi::Tensor neuron_scale, ::tyloi::Tensor codebook, ::tyloi::Tensor x,
    ::tyloi::Tensor ids, ::tyloi::Tensor expert_local, int64_t n_experts,
    int64_t pool_experts, int64_t out_per_expert, int64_t neuron_len,
    int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode,
    bool input_quantized, ::tyloi::Tensor out, ::tyloi::Tensor qx,
    ::tyloi::Tensor xscale, ::tyloi::Tensor ids_dst, ::tyloi::Tensor expert_bounds,
    ::tyloi::Tensor tile_bounds, ::tyloi::Tensor tile_experts);
::tyloi::Tensor nvq_moe_grouped_matmul_pool_f16_cuda(
    ::tyloi::Tensor indices, ::tyloi::Tensor aux, ::tyloi::Tensor sub_scale,
    ::tyloi::Tensor neuron_scale, ::tyloi::Tensor codebook, ::tyloi::Tensor x,
    ::tyloi::Tensor expert_local, int64_t n_experts, int64_t pool_experts,
    int64_t out_per_expert, int64_t neuron_len, int64_t gs,
    int64_t sub_bits, int64_t format, int64_t sign_mode,
    ::tyloi::Tensor out, ::tyloi::Tensor ids_dst, ::tyloi::Tensor expert_bounds,
    ::tyloi::Tensor tile_bounds, ::tyloi::Tensor tile_experts);
::tyloi::Tensor nvq_moe_grouped_matmul_hetero_f16_cuda(
    ::tyloi::Tensor weight_ptrs, ::tyloi::Tensor weight_sizes,
    ::tyloi::Tensor pool_params, ::tyloi::Tensor expert_pool,
    ::tyloi::Tensor expert_local, ::tyloi::Tensor x,
    int64_t n_experts, int64_t out_per_expert, int64_t neuron_len,
    int64_t route_tile_m, ::tyloi::Tensor out,
    ::tyloi::Tensor ids_dst,
    ::tyloi::Tensor expert_bounds, ::tyloi::Tensor tile_bounds,
    ::tyloi::Tensor tile_experts, int64_t format_group,
    bool masked_experts);
::tyloi::Tensor nvq_moe_grouped_matmul_hetero_ws_cuda(
    ::tyloi::Tensor weight_ptrs, ::tyloi::Tensor weight_sizes,
    ::tyloi::Tensor pool_params, ::tyloi::Tensor expert_pool,
    ::tyloi::Tensor expert_local, ::tyloi::Tensor x,
    ::tyloi::Tensor ids, int64_t n_experts, int64_t out_per_expert,
    int64_t neuron_len, bool input_quantized, ::tyloi::Tensor out,
    ::tyloi::Tensor qx, ::tyloi::Tensor xscale);
::tyloi::Tensor nvq_moe_grouped_matmul_active_ws_cuda(
    ::tyloi::Tensor weight_ptrs, ::tyloi::Tensor weight_sizes,
    ::tyloi::Tensor pool_params, ::tyloi::Tensor expert_pool,
    ::tyloi::Tensor expert_local, ::tyloi::Tensor x,
    ::tyloi::Tensor ids, int64_t n_experts, int64_t out_per_expert,
    int64_t neuron_len, bool input_quantized, ::tyloi::Tensor out,
    ::tyloi::Tensor qx, ::tyloi::Tensor xscale,
    ::tyloi::Tensor active_plan);
::tyloi::Tensor nepq_moe_grouped_matmul_pool_ws_cuda(
    ::tyloi::Tensor indices, ::tyloi::Tensor aux, ::tyloi::Tensor sub_scale,
    ::tyloi::Tensor neuron_scale, ::tyloi::Tensor table_pool, ::tyloi::Tensor bank_ids,
    ::tyloi::Tensor grouped_table_pool, ::tyloi::Tensor x, ::tyloi::Tensor ids,
    ::tyloi::Tensor expert_local, int64_t n_experts, int64_t pool_experts,
    int64_t out_per_expert, int64_t neuron_len, int64_t sub_bits, int64_t format,
    bool input_quantized, ::tyloi::Tensor out, ::tyloi::Tensor qx,
    ::tyloi::Tensor xscale, ::tyloi::Tensor ids_dst, ::tyloi::Tensor expert_bounds,
    ::tyloi::Tensor tile_bounds, ::tyloi::Tensor tile_experts);
::tyloi::Tensor nepq_moe_grouped_matmul_pool_f16_cuda(
    ::tyloi::Tensor indices, ::tyloi::Tensor aux, ::tyloi::Tensor sub_scale,
    ::tyloi::Tensor neuron_scale, ::tyloi::Tensor table_pool,
    ::tyloi::Tensor bank_ids, ::tyloi::Tensor x, ::tyloi::Tensor expert_local,
    int64_t n_experts, int64_t pool_experts, int64_t out_per_expert,
    int64_t neuron_len, int64_t sub_bits, int64_t format,
    ::tyloi::Tensor out, ::tyloi::Tensor ids_dst, ::tyloi::Tensor expert_bounds,
    ::tyloi::Tensor tile_bounds, ::tyloi::Tensor tile_experts);
::tyloi::Tensor nepq_sparse_residual_grouped_cuda(
    ::tyloi::Tensor dictionary, ::tyloi::Tensor first, ::tyloi::Tensor second,
    ::tyloi::Tensor input, ::tyloi::Tensor route_ids, ::tyloi::Tensor expert_local,
    int64_t out_per_expert, int64_t position_bits, int64_t block_vectors,
    ::tyloi::Tensor output);
::tyloi::Tensor moe_weighted_reduce_cuda(::tyloi::Tensor pair_output, ::tyloi::Tensor weights);
::tyloi::Tensor moe_swiglu_split_cuda(::tyloi::Tensor gate_up);
::tyloi::Tensor moe_swiglu_rounded_cuda(
    ::tyloi::Tensor gate, ::tyloi::Tensor up,
    ::tyloi::Tensor sigmoid_table);
::tyloi::Tensor moe_swiglu_sigmoid_table_cuda();
::tyloi::Tensor moe_geglu_split_cuda(::tyloi::Tensor gate_up);
::tyloi::Tensor moe_apply_expert_scale_cuda(
    ::tyloi::Tensor weights, ::tyloi::Tensor ids, ::tyloi::Tensor scales);
::tyloi::Tensor moe_add_shared_gate_cuda(
    ::tyloi::Tensor routed, ::tyloi::Tensor shared, ::tyloi::Tensor gate_logits);
::tyloi::Tensor moe_weighted_reduce_shared_gate_cuda(
    ::tyloi::Tensor pair_output, ::tyloi::Tensor weights,
    ::tyloi::Tensor shared, ::tyloi::Tensor gate_logits);
::tyloi::Tensor mxfp4_moe_grouped_matmul_pool_f16_cuda(
    ::tyloi::Tensor values, ::tyloi::Tensor scales, ::tyloi::Tensor input,
    ::tyloi::Tensor ids, ::tyloi::Tensor expert_local,
    int64_t global_experts, int64_t pool_experts,
    int64_t out_per_expert, int64_t neuron_len,
    ::tyloi::Tensor output, ::tyloi::Tensor ids_dst,
    ::tyloi::Tensor expert_bounds, ::tyloi::Tensor tile_bounds,
    ::tyloi::Tensor tile_experts);
