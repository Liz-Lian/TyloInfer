#pragma once
#include <torch/extension.h>
#include <optional>
#include <vector>

namespace mfq_qwen4_jit {
/** Normalize independent groups with an offset weight. */
torch::Tensor grouped_rms_norm(const torch::Tensor&,const torch::Tensor&,int64_t,double);
/** Prepare residual streams and the optional injection gate. */
std::vector<torch::Tensor> gated_residual_pre(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const std::optional<torch::Tensor>&,int64_t,int64_t,double);
/** Apply the gated update to residual streams. */
torch::Tensor gated_residual_post(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,int64_t);
/** Compute pooled index scores. */
torch::Tensor block_scores(const torch::Tensor&,const torch::Tensor&);
/** Apply dilated convolution and update the PLE state. */
std::vector<torch::Tensor> ple_dilated_conv_silu(const torch::Tensor&,const torch::Tensor&,const std::optional<torch::Tensor>&,int64_t);
/** Compute causal dense grouped-query attention. */
torch::Tensor dense_gqa_attention(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,int64_t);
/** Compute selected-token grouped-query attention. */
torch::Tensor sparse_gqa_attention(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);
}
