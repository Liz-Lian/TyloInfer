#pragma once

#include "moe_types.h"

namespace mfq::cuda {

struct MfeScratch {
    std::unordered_map<MoeActivationKey, MoeActivationWorkspace, MoeActivationKeyHash> values;

    MoeActivationWorkspace get(mfq_tensor_backend::Tensor input, int rows, int groups,
                               int gs, bool stable = true);
};

// Borrowed NINT expert pools; the caller retains scratch and any cache leases.
struct Mfe {
    int n_experts;
    int out_per_expert;
    int neuron_len;
    bool partial_experts;
    const std::vector<MfePoolWeight>& pools;

    mfq_tensor_backend::Tensor forward(
        MfeScratch& scratch, mfq_tensor_backend::Tensor input, const MoeRoutePlan& route,
        bool prequantized = false, int epilogue = 0) const;
    mfq_tensor_backend::Tensor clamped(
        MfeScratch& scratch, mfq_tensor_backend::Tensor input, const MoeRoutePlan& route,
        double limit) const;
};

} // namespace mfq::cuda
