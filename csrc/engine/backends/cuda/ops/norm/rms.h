#pragma once

#include "backend.h"

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/tensor.h"
#include "engine/ops.h"
namespace mfq::cuda::ops {
mfq::Tensor rms_norm(mfq::Tensor input, const mfq::Tensor& weight, double eps, double offset,
                     mfq::ScalarType output_type);
std::array<mfq::Tensor, 3> qkv_norm(std::array<mfq::Tensor, 3> values,
    std::array<mfq::Tensor, 3> weights, double eps, double offset, mfq::ScalarType output_type);
mfq::ops::ResidualNorms residual_norms(mfq::Tensor residual, mfq::Tensor attention,
    const mfq::Tensor& attention_weight, const mfq::Tensor& dense_weight,
    const mfq::Tensor& router_weight, const mfq::Tensor& expert_weight, double eps);
mfq::Tensor merge_norms(mfq::Tensor dense, mfq::Tensor experts, mfq::Tensor residual,
    const mfq::Tensor& dense_weight, const mfq::Tensor& expert_weight,
    const mfq::Tensor& output_weight, const mfq::Tensor& scale, double eps);
}
#endif

mfq_tensor_backend::Tensor qwen_rms_norm(
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor weight,
    double eps,
    double weight_offset);

mfq_tensor_backend::Tensor qwen_rms_norm_bf16(
    bool fused,
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor weight,
    double eps,
    double weight_offset);
