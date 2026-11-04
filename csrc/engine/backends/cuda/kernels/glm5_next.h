#pragma once

#include "backend.h"

#include <optional>
#include <vector>

namespace mfq_glm5_next {
using ::tyloi::DenseTensor;

std::vector<DenseTensor> mhc_pre(
    const DenseTensor&, const DenseTensor&, const DenseTensor&, const DenseTensor&,
    int64_t sinkhorn_iterations, double hc_eps, double rms_eps);
DenseTensor mhc_post(const DenseTensor&, const DenseTensor&, const DenseTensor&, const DenseTensor&);
DenseTensor kda_forget_gate(
    const DenseTensor&, const DenseTensor&, const DenseTensor&, const DenseTensor&, const DenseTensor&,
    int64_t num_heads, int64_t head_dim, double lower_bound);
DenseTensor kpool_scores(const DenseTensor&, const DenseTensor&, const DenseTensor&);
DenseTensor kpool_states(
    const DenseTensor&, const DenseTensor&, const DenseTensor&, int64_t pool);
DenseTensor dense_mla_attention(
    const DenseTensor&, const DenseTensor&, int64_t offset,
    std::optional<double> scale = std::nullopt);
DenseTensor sparse_mla_attention(
    const DenseTensor&, const DenseTensor&, const DenseTensor& indices,
    std::optional<double> scale = std::nullopt);

} // namespace mfq_glm5_next
