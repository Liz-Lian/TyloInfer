#pragma once

#include "backend.h"

#include <optional>
#include <vector>

namespace mfq_glm5_next {
using ::tyloi::DenseTensor;

/** Normalize residual streams and compute their Sinkhorn mixing and collapsed input. */
std::vector<DenseTensor> mhc_pre(
    const DenseTensor&, const DenseTensor&, const DenseTensor&, const DenseTensor&,
    int64_t sinkhorn_iterations, double hc_eps, double rms_eps);
/** Mix the residual streams with the projected mHC branch. */
DenseTensor mhc_post(const DenseTensor&, const DenseTensor&, const DenseTensor&, const DenseTensor&);
/** Compute bounded per-head KDA forget gates from two projections. */
DenseTensor kda_forget_gate(
    const DenseTensor&, const DenseTensor&, const DenseTensor&, const DenseTensor&, const DenseTensor&,
    int64_t num_heads, int64_t head_dim, double lower_bound);
/** Score pooled keys with normalized per-head weights. */
DenseTensor kpool_scores(const DenseTensor&, const DenseTensor&, const DenseTensor&);
/** Pool complete key blocks using gated softmax weights. */
DenseTensor kpool_states(
    const DenseTensor&, const DenseTensor&, const DenseTensor&, int64_t pool);
/** Run causal MLA over the complete retained latent cache. */
DenseTensor dense_mla_attention(
    const DenseTensor&, const DenseTensor&, int64_t offset,
    std::optional<double> scale = std::nullopt);
/** Run MLA over the caller's selected latent cache rows. */
DenseTensor sparse_mla_attention(
    const DenseTensor&, const DenseTensor&, const DenseTensor& indices,
    std::optional<double> scale = std::nullopt);

} // namespace mfq_glm5_next
