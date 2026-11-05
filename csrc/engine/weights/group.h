#pragma once

#include "engine/backends/cuda/ops/projection/projection.h"

namespace tyloi::cuda {

#if TYLOINFER_ENABLE_MFQ_COMPAT
std::vector<std::shared_ptr<const ::tyloi::Tensor>> nint_aliases(
    const std::vector<std::shared_ptr<const ::tyloi::Tensor>>& weights,
    const NintLinearGroup& group);
#endif

Projections make_projections(
    std::vector<std::shared_ptr<const ::tyloi::Tensor>> weights,
    bool group_nint,
    bool cpu_layer,
    size_t prefix = 1);

}

DenseLinearGroup make_fp32_quant_group(
    tyloi::cuda::Collectives& collectives,
    int primary,
    QuantLinearGroup group);

QuantLinearGroup make_quant_group(
    std::vector<std::shared_ptr<const ::tyloi::Tensor>> layers,
    bool group_nint,
    bool cpu_layer,
    bool preserve_projection_boundaries = false);

DenseLinearGroup make_dense_group(
    const std::vector<::tyloi::DenseTensor>& weights);
