#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

::tyloi::DenseTensor rms_norm_cuda(::tyloi::DenseTensor x, ::tyloi::DenseTensor weight, double eps);
::tyloi::DenseTensor rms_norm_offset_cuda(::tyloi::DenseTensor x, ::tyloi::DenseTensor weight, double eps, double weight_offset);
::tyloi::DenseTensor rms_norm_f16_cuda(::tyloi::DenseTensor x, ::tyloi::DenseTensor weight, double eps,
                                double weight_offset);
::tyloi::DenseTensor grouped_rms_norm_cuda(::tyloi::DenseTensor x,
    ::tyloi::DenseTensor weight, int64_t group, double eps, double weight_offset);
// Grouped gated RMSNorm: y = weight * (x * silu(gate)) / sqrt(mean_group(hat^2) + eps),
// i.e. the gate is applied before the group mean square. x/gate are
// [..., D] contiguous CUDA tensors of any f32/f16/bf16 dtype (each may differ);
// group_size divides D; the output keeps the dtype of x.
::tyloi::DenseTensor grouped_gated_rms_norm_cuda(
    ::tyloi::DenseTensor x, ::tyloi::DenseTensor gate,
    ::tyloi::DenseTensor weight, int64_t group_size, double eps);
::tyloi::DenseTensor gdn_rms_norm_gate_cuda(::tyloi::DenseTensor attended,
    ::tyloi::DenseTensor gate, ::tyloi::DenseTensor weight, double eps,
    bool silu_gate, bool output_half);
::tyloi::DenseTensor qwen_rms_norm_bf16_cuda(
    ::tyloi::DenseTensor input, ::tyloi::DenseTensor weight, double eps,
    double weight_offset);
std::vector<::tyloi::DenseTensor> qwen_rms_norm_pair_bf16_cuda(
    ::tyloi::DenseTensor first, ::tyloi::DenseTensor second,
    ::tyloi::DenseTensor first_weight, ::tyloi::DenseTensor second_weight,
    double eps, double weight_offset);
std::vector<::tyloi::DenseTensor> rms_norm_pair_f16_f32_offset_cuda(
    ::tyloi::DenseTensor first, ::tyloi::DenseTensor second,
    ::tyloi::DenseTensor first_weight, ::tyloi::DenseTensor second_weight,
    double eps, double weight_offset);
::tyloi::DenseTensor l2_norm_cuda(::tyloi::DenseTensor x, double eps);
::tyloi::DenseTensor acc_cuda(::tyloi::DenseTensor a, ::tyloi::DenseTensor b);
std::vector<::tyloi::DenseTensor> acc_rms_norm_cuda(::tyloi::DenseTensor a, ::tyloi::DenseTensor b,
                                             ::tyloi::DenseTensor weight, double eps,
                                             double weight_offset);
std::vector<::tyloi::DenseTensor> acc_rms_norm_f16_cuda(::tyloi::DenseTensor a, ::tyloi::DenseTensor b,
                                                 ::tyloi::DenseTensor weight, double eps,
                                                 double weight_offset);
std::vector<::tyloi::DenseTensor> acc_rms_norm_bf16_cuda(
    ::tyloi::DenseTensor a, ::tyloi::DenseTensor b,
    ::tyloi::DenseTensor weight, double eps, double weight_offset);
std::vector<::tyloi::DenseTensor> gemma4_attn_residual_pre_norms_f16_cuda(
    ::tyloi::DenseTensor residual, ::tyloi::DenseTensor attn,
    ::tyloi::DenseTensor attn_post_weight, ::tyloi::DenseTensor dense_pre_weight,
    ::tyloi::DenseTensor router_weight, ::tyloi::DenseTensor moe_pre_weight, double eps);
::tyloi::DenseTensor gemma4_ffn_merge_f16_cuda(
    ::tyloi::DenseTensor dense, ::tyloi::DenseTensor moe, ::tyloi::DenseTensor residual,
    ::tyloi::DenseTensor dense_post_weight, ::tyloi::DenseTensor moe_post_weight,
    ::tyloi::DenseTensor final_post_weight, ::tyloi::DenseTensor layer_scale, double eps);
::tyloi::DenseTensor glm_dsa_indexer_layer_norm_cuda(
    ::tyloi::DenseTensor x, ::tyloi::DenseTensor weight, ::tyloi::DenseTensor bias, double eps);
