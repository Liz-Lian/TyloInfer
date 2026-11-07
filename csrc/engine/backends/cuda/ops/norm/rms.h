#pragma once

#include "backend.h"

#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/tensor.h"
#include "engine/ops.h"
namespace tyloi::cuda::ops {
tyloi::DenseTensor rms_norm(tyloi::DenseTensor input, const tyloi::DenseTensor& weight, double eps, double offset,
                     tyloi::ScalarType output_type);
std::array<tyloi::DenseTensor, 3> qkv_norm(std::array<tyloi::DenseTensor, 3> values,
    std::array<const tyloi::DenseTensor*, 3> weights, double eps, double offset, tyloi::ScalarType output_type);
std::optional<tyloi::ops::ResidualNorms> residual_norms(tyloi::DenseTensor residual, tyloi::DenseTensor attention,
    const tyloi::DenseTensor& attention_weight, const tyloi::DenseTensor& dense_weight,
    const tyloi::DenseTensor& router_weight, const tyloi::DenseTensor& expert_weight, double eps);
tyloi::DenseTensor merge_norms(tyloi::DenseTensor dense, tyloi::DenseTensor experts, tyloi::DenseTensor residual,
    const tyloi::DenseTensor& dense_weight, const tyloi::DenseTensor& expert_weight,
    const tyloi::DenseTensor& output_weight, const tyloi::DenseTensor& scale, double eps);
}
#endif

::tyloi::DenseTensor qwen_rms_norm(
    ::tyloi::DenseTensor x,
#ifdef TYLOINFER_NATIVE_CUDA
    const ::tyloi::Tensor& weight,
#else
    ::tyloi::DenseTensor weight,
#endif
    double eps,
    double weight_offset);

::tyloi::DenseTensor qwen_rms_norm_bf16(
    bool fused,
    ::tyloi::DenseTensor x,
    ::tyloi::DenseTensor weight,
    double eps,
    double weight_offset);
