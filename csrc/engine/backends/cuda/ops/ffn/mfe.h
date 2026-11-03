#pragma once

#include "engine/weights/moe/pool.h"
#include "engine/backends/cuda/ops/ffn/activation.h"
#include "engine/backends/cuda/ops/ffn/routes.h"

namespace mfq::cuda {

// Borrowed NINT expert pools; the caller retains scratch and any cache leases.
struct Mfe {
    int n_experts;
    int out_per_expert;
    int neuron_len;
    bool partial_experts;
    const std::shared_ptr<const std::vector<MfePoolWeight>>& pools;

    mfq_tensor_backend::Tensor forward(
        MfeScratch& scratch, mfq_tensor_backend::Tensor input, const MoeRoutePlan& route,
        bool prequantized = false, int epilogue = 0) const;
    mfq_tensor_backend::Tensor clamped(
        MfeScratch& scratch, mfq_tensor_backend::Tensor input, const MoeRoutePlan& route,
        double limit) const;
};

} // namespace mfq::cuda
