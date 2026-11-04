#pragma once

#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/ops.h"

#include <cstddef>

// Default bound on the expert matrices dense_routed_projection gathers at once.
inline constexpr std::size_t kDenseRoutedGatherBudgetBytes = std::size_t(512) << 20;

// Apply a dense [experts,out,in] routed bank to every route of every token.
// `input` is [tokens,in] or [tokens,routes,in]; `ids` is [tokens,routes]. The
// bank keeps its source precision. Each route gathers its whole expert matrix,
// so tokens run in chunks whose gathered matrices fit `gather_budget_bytes`.
::tyloi::Tensor dense_routed_projection(
    const ::tyloi::Tensor &bank, const ::tyloi::Tensor &input,
    const ::tyloi::Tensor &ids,
    std::size_t gather_budget_bytes = kDenseRoutedGatherBudgetBytes);

namespace tyloi::cuda {

::tyloi::Tensor expert_activation(
    tyloi::ops::MlpActivation activation, ::tyloi::Tensor gate,
    ::tyloi::Tensor up, double limit);

// Read-only projection view; routing, cache leases and branch scheduling stay with the caller.
struct Mlp {
    const QuantLinearGroup& gate_up;
    const QuantLinear& down;
    tyloi::ops::MlpActivation activation;
    bool geglu;
    double swiglu_limit;

    struct Options {
        bool nvq_fusion = true;
        bool gelu_fusion = true;
        bool silu_fusion = true;
    };

    void check(const ::tyloi::Tensor& input, int primary) const;
    bool parallel() const;
    ::tyloi::Tensor sharded(
        CudaProfiler& profiler, KlMmqState& mmq, TqExec& tq, Collectives& collectives,
        int primary, bool peer_first, bool reduce_to_primary, bool fp16_reduce,
        ::tyloi::Tensor input) const;
    ::tyloi::Tensor forward(
        const Projector& project, CudaProfiler& profiler, const Options& options,
        ::tyloi::Tensor input) const;
    ::tyloi::Tensor bf16(
        const Projector& project, CudaProfiler& profiler, bool fused,
        ::tyloi::Tensor input) const;
    ::tyloi::Tensor kld(
        const Projector& project, CudaProfiler& profiler, KlMmqState& mmq,
        ::tyloi::Tensor input) const;
    bool can_fuse(const ::tyloi::Tensor& input,
                  const ::tyloi::Tensor& residual) const;
    ::tyloi::Tensor fused(
        CudaProfiler& profiler, ::tyloi::Tensor input,
        ::tyloi::Tensor residual) const;
};

} // namespace tyloi::cuda
