#pragma once

#include "ops/common/include/projection.h"
#include "engine/models/common/gated_mlp.h"

#include <cstddef>

// Default bound on the expert matrices dense_routed_projection gathers at once.
inline constexpr std::size_t kDenseRoutedGatherBudgetBytes = std::size_t(512) << 20;

// Apply a dense [experts,out,in] routed bank to every route of every token.
// `input` is [tokens,in] or [tokens,routes,in]; `ids` is [tokens,routes]. The
// bank keeps its source precision. Each route gathers its whole expert matrix,
// so tokens run in chunks whose gathered matrices fit `gather_budget_bytes`.
mfq_tensor_backend::Tensor dense_routed_projection(
    const mfq_tensor_backend::Tensor &bank, const mfq_tensor_backend::Tensor &input,
    const mfq_tensor_backend::Tensor &ids,
    std::size_t gather_budget_bytes = kDenseRoutedGatherBudgetBytes);

namespace mfq::cuda {

mfq_tensor_backend::Tensor expert_activation(
    mfq::models::MlpActivation activation, mfq_tensor_backend::Tensor gate,
    mfq_tensor_backend::Tensor up, double limit);

// Read-only projection view; routing, cache leases and branch scheduling stay with the caller.
struct Mlp {
    const QuantLinearGroup& gate_up;
    const QuantLinear& down;
    mfq::models::MlpActivation activation;
    bool geglu;
    double swiglu_limit;

    struct Options {
        bool nvq_fusion = true;
        bool gelu_fusion = true;
        bool silu_fusion = true;
    };

    bool parallel() const;
    mfq_tensor_backend::Tensor sharded(
        CudaProfiler& profiler, KlMmqState& mmq, TqExec& tq, Collectives& collectives,
        int primary, bool peer_first, bool reduce_to_primary, bool fp16_reduce,
        mfq_tensor_backend::Tensor input) const;
    mfq_tensor_backend::Tensor forward(
        const Projector& project, CudaProfiler& profiler, const Options& options,
        mfq_tensor_backend::Tensor input) const;
    mfq_tensor_backend::Tensor bf16(
        const Projector& project, CudaProfiler& profiler, bool fused,
        mfq_tensor_backend::Tensor input) const;
    mfq_tensor_backend::Tensor kld(
        const Projector& project, CudaProfiler& profiler, KlMmqState& mmq,
        mfq_tensor_backend::Tensor input) const;
    bool can_fuse(const mfq_tensor_backend::Tensor& input,
                  const mfq_tensor_backend::Tensor& residual) const;
    mfq_tensor_backend::Tensor fused(
        CudaProfiler& profiler, mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor residual) const;
};

} // namespace mfq::cuda
