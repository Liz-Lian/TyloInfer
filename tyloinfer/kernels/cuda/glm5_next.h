#pragma once

#include <torch/extension.h>

#include <optional>
#include <vector>

namespace mfq_glm5_jit {

/** Normalize residual streams and compute their Sinkhorn mixing and collapsed input. */
std::vector<torch::Tensor> mhc_pre(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    int64_t sinkhorn_iterations, double hc_eps, double rms_eps);
/** Mix the residual streams with the projected mHC branch. */
torch::Tensor mhc_post(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
/** Compute bounded per-head KDA forget gates from two projections. */
torch::Tensor kda_forget_gate(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    int64_t num_heads, int64_t head_dim, double lower_bound);
/** Score pooled keys with normalized per-head weights. */
torch::Tensor kpool_scores(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&);
/** Pool complete key blocks using gated softmax weights. */
torch::Tensor kpool_states(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, int64_t pool);
/** Run causal MLA over the complete retained latent cache. */
torch::Tensor dense_mla_attention(
    const torch::Tensor&, const torch::Tensor&, int64_t offset,
    std::optional<double> scale = std::nullopt);
/** Run MLA over the caller's selected latent cache rows. */
torch::Tensor sparse_mla_attention(
    const torch::Tensor&, const torch::Tensor&, const torch::Tensor& indices,
    std::optional<double> scale = std::nullopt);

} // namespace mfq_glm5_jit
