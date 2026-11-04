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
::tyloi::DenseTensor dense_routed_projection(
    const ::tyloi::DenseTensor &bank, const ::tyloi::DenseTensor &input,
    const ::tyloi::DenseTensor &ids,
    std::size_t gather_budget_bytes = kDenseRoutedGatherBudgetBytes);

namespace tyloi::cuda {

::tyloi::DenseTensor expert_activation(
    tyloi::ops::MlpActivation activation, ::tyloi::DenseTensor gate,
    ::tyloi::DenseTensor up, double limit);

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

    void check(const ::tyloi::DenseTensor& input, int primary) const;
    bool parallel() const;
    ::tyloi::DenseTensor sharded(
        CudaProfiler& profiler, KlMmqState& mmq, TqExec& tq, Collectives& collectives,
        int primary, bool peer_first, bool reduce_to_primary, bool fp16_reduce,
        ::tyloi::DenseTensor input) const;
    ::tyloi::DenseTensor forward(
        const Projector& project, CudaProfiler& profiler, const Options& options,
        ::tyloi::DenseTensor input) const;
    ::tyloi::DenseTensor bf16(
        const Projector& project, CudaProfiler& profiler, bool fused,
        ::tyloi::DenseTensor input) const;
    ::tyloi::DenseTensor kld(
        const Projector& project, CudaProfiler& profiler, KlMmqState& mmq,
        ::tyloi::DenseTensor input) const;
    bool can_fuse(const ::tyloi::DenseTensor& input,
                  const ::tyloi::DenseTensor& residual) const;
    ::tyloi::DenseTensor fused(
        CudaProfiler& profiler, ::tyloi::DenseTensor input,
        ::tyloi::DenseTensor residual) const;
};

} // namespace tyloi::cuda
