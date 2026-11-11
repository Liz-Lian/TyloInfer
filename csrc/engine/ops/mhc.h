#pragma once
#include "engine/tensor.h"
#include <cstdint>

namespace tyloi::ops {
/** Numerical policy for multi-head connection mixing. */
struct MhcConfig {
    std::int64_t iterations = 0;
    double rms_eps = 0, coefficient_eps = 0;
};
/** Own the collapsed branch and its residual mixing coefficients. */
struct MhcResult { Tensor branch, next_pre, post, combination; };
/** Collapse a residual stream and produce coefficients for its later expansion. */
MhcResult mhc_collapse(const Tensor& residual, const Tensor& previous_pre, const Tensor& function,
    const Tensor& scale, const Tensor& base, const Tensor& norm, MhcConfig config);
/** Compute the normalized input for a source-only attention stage. */
Tensor mhc_input(const Tensor& residual, const Tensor& previous_pre, const Tensor& norm, double rms_eps);
/** Expand a computed branch back into its original residual streams. */
Tensor mhc_expand(const Tensor& branch, const Tensor& residual, const MhcResult& mix);
} // namespace tyloi::ops
