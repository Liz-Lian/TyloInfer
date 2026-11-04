#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

::tyloi::Tensor rms_norm_cuda(::tyloi::Tensor x, ::tyloi::Tensor weight, double eps);
::tyloi::Tensor rms_norm_offset_cuda(::tyloi::Tensor x, ::tyloi::Tensor weight, double eps, double weight_offset);
::tyloi::Tensor rms_norm_f16_cuda(::tyloi::Tensor x, ::tyloi::Tensor weight, double eps,
                                double weight_offset);
::tyloi::Tensor grouped_rms_norm_cuda(::tyloi::Tensor x,
    ::tyloi::Tensor weight, int64_t group, double eps, double weight_offset);
// Grouped gated RMSNorm: y = weight * (x * silu(gate)) / sqrt(mean_group(hat^2) + eps),
// i.e. the gate is applied before the group mean square. x/gate are
// [..., D] contiguous CUDA tensors of any f32/f16/bf16 dtype (each may differ);
// group_size divides D; the output keeps the dtype of x.
::tyloi::Tensor grouped_gated_rms_norm_cuda(
    ::tyloi::Tensor x, ::tyloi::Tensor gate,
    ::tyloi::Tensor weight, int64_t group_size, double eps);
::tyloi::Tensor gdn_rms_norm_gate_cuda(::tyloi::Tensor attended,
    ::tyloi::Tensor gate, ::tyloi::Tensor weight, double eps,
    bool silu_gate, bool output_half);
::tyloi::Tensor qwen_rms_norm_bf16_cuda(
    ::tyloi::Tensor input, ::tyloi::Tensor weight, double eps,
    double weight_offset);
std::vector<::tyloi::Tensor> qwen_rms_norm_pair_bf16_cuda(
    ::tyloi::Tensor first, ::tyloi::Tensor second,
    ::tyloi::Tensor first_weight, ::tyloi::Tensor second_weight,
    double eps, double weight_offset);
std::vector<::tyloi::Tensor> rms_norm_pair_f16_f32_offset_cuda(
    ::tyloi::Tensor first, ::tyloi::Tensor second,
    ::tyloi::Tensor first_weight, ::tyloi::Tensor second_weight,
    double eps, double weight_offset);
::tyloi::Tensor l2_norm_cuda(::tyloi::Tensor x, double eps);
::tyloi::Tensor acc_cuda(::tyloi::Tensor a, ::tyloi::Tensor b);
std::vector<::tyloi::Tensor> acc_rms_norm_cuda(::tyloi::Tensor a, ::tyloi::Tensor b,
                                             ::tyloi::Tensor weight, double eps,
                                             double weight_offset);
std::vector<::tyloi::Tensor> acc_rms_norm_f16_cuda(::tyloi::Tensor a, ::tyloi::Tensor b,
                                                 ::tyloi::Tensor weight, double eps,
                                                 double weight_offset);
std::vector<::tyloi::Tensor> acc_rms_norm_bf16_cuda(
    ::tyloi::Tensor a, ::tyloi::Tensor b,
    ::tyloi::Tensor weight, double eps, double weight_offset);
std::vector<::tyloi::Tensor> gemma4_attn_residual_pre_norms_f16_cuda(
    ::tyloi::Tensor residual, ::tyloi::Tensor attn,
    ::tyloi::Tensor attn_post_weight, ::tyloi::Tensor dense_pre_weight,
    ::tyloi::Tensor router_weight, ::tyloi::Tensor moe_pre_weight, double eps);
::tyloi::Tensor gemma4_ffn_merge_f16_cuda(
    ::tyloi::Tensor dense, ::tyloi::Tensor moe, ::tyloi::Tensor residual,
    ::tyloi::Tensor dense_post_weight, ::tyloi::Tensor moe_post_weight,
    ::tyloi::Tensor final_post_weight, ::tyloi::Tensor layer_scale, double eps);
::tyloi::Tensor glm_dsa_indexer_layer_norm_cuda(
    ::tyloi::Tensor x, ::tyloi::Tensor weight, ::tyloi::Tensor bias, double eps);
