#pragma once

#include <optional>

namespace tyloi::cuda {

/** Numerical policy for gated residual projections and fusion. */
struct ResidualOptions {
    bool fused_projections = true;
    bool fused_activation = true;
    bool fused_post_norm = true;
    bool prepared_dense = false;
    bool native_input = true;
    bool fixed_group = true;
    bool compact_mix = true;
    bool two_stage = true;
    bool dense_injection = true;
};

struct FfnOptions {
    bool small_heterogeneous = true;
    bool delayed_readback = true;
    bool bundle_prefetch = true;
    bool swiglu_fusion = true;
    bool reduce_fusion = true;
};

struct FfnMode {
    bool materialized = false;
    bool unfused = false;
    bool serial = false;
    bool batch_serial = false;
    int tile = 0;
    int tiles = 1;
    bool reuse_input = true;
    bool projection_major = false;
    bool independent = false;
    // Scoped diagnostic overrides; absent values retain the executor policy.
    std::optional<bool> moe_prefill;
    std::optional<bool> moe_nvq_decode;
    std::optional<bool> moe_nint_decode;
};

} // namespace tyloi::cuda
