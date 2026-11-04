#pragma once

#include "backend.h"

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/tensor.h"
#include "engine/ops.h"
namespace tyloi::cuda::ops {
tyloi::Tensor rms_norm(tyloi::Tensor input, const tyloi::Tensor& weight, double eps, double offset,
                     tyloi::ScalarType output_type);
std::array<tyloi::Tensor, 3> qkv_norm(std::array<tyloi::Tensor, 3> values,
    std::array<tyloi::Tensor, 3> weights, double eps, double offset, tyloi::ScalarType output_type);
std::optional<tyloi::ops::ResidualNorms> residual_norms(tyloi::Tensor residual, tyloi::Tensor attention,
    const tyloi::Tensor& attention_weight, const tyloi::Tensor& dense_weight,
    const tyloi::Tensor& router_weight, const tyloi::Tensor& expert_weight, double eps);
tyloi::Tensor merge_norms(tyloi::Tensor dense, tyloi::Tensor experts, tyloi::Tensor residual,
    const tyloi::Tensor& dense_weight, const tyloi::Tensor& expert_weight,
    const tyloi::Tensor& output_weight, const tyloi::Tensor& scale, double eps);
}
#endif

::tyloi::Tensor qwen_rms_norm(
    ::tyloi::Tensor x,
    ::tyloi::Tensor weight,
    double eps,
    double weight_offset);

::tyloi::Tensor qwen_rms_norm_bf16(
    bool fused,
    ::tyloi::Tensor x,
    ::tyloi::Tensor weight,
    double eps,
    double weight_offset);
