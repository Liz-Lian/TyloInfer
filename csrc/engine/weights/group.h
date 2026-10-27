#pragma once

#include "ops/common/include/quant_linear.h"

DenseLinearGroup make_fp32_quant_group(
    mfq::cuda::Collectives& collectives,
    int primary,
    QuantLinearGroup group);

QuantLinearGroup make_quant_group(
    std::vector<QuantLinear> layers,
    bool group_nint,
    bool cpu_layer,
    bool preserve_projection_boundaries = false);

DenseLinearGroup make_dense_group(
    const std::vector<mfq_tensor_backend::Tensor>& weights);
