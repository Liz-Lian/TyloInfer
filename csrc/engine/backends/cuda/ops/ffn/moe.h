#pragma once

#include "engine/weights/moe/pool.h"
#include "engine/weights/moe/dispatch.h"
#include "engine/backends/cuda/ops/ffn/activation.h"
#include "engine/backends/cuda/ops/ffn/routes.h"
#include "engine/backends/cuda/ops/projection/mmq.h"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

void fp8_sq_moe_matmul(
    const tyloi::Fp8SqTensor& weight,
    ::tyloi::DenseTensor input,
    ::tyloi::DenseTensor expert_ids,
    ::tyloi::DenseTensor expert_local,
    int n_experts,
    int local_experts,
    int out_per_expert,
    int neuron_len,
    ::tyloi::DenseTensor output);

struct MixedNintInputPlan {
    ::tyloi::DenseTensor weight_ptrs,quantize_descriptors;
    ::tyloi::DenseTensor row_metadata_workspace;
    int total_groups=0;
};

struct MoeOptions {
    bool prefill = true;
    int min_tokens = 9;
    bool nvq_decode = true;
    bool active_decode = false;
    bool nint_decode = true;
    std::optional<bool> row_workspace, direct_pool, group_dot, route_hint;
    std::optional<int> route_warps;
    bool heterogeneous = true;
};

struct MoeScratch {
    std::unordered_map<int,MixedNintInputPlan> nint_input_plans;
    std::unordered_map<int,::tyloi::DenseTensor> nvq_active_plans;
    std::unordered_map<MixedMoeActivationKey, MoeActivationWorkspace,
                       MixedMoeActivationKeyHash> activation_workspaces;
    ::tyloi::DenseTensor nint_expert_pool, nint_expert_local;
    ::tyloi::DenseTensor nvq_expert_pool, nvq_expert_local;
    std::weak_ptr<MixedNintDispatch> nint_binding;
    std::weak_ptr<MixedNvqDispatch> nvq_binding;

    void bind(const std::shared_ptr<MixedNintDispatch>& dispatch);
    void bind(const std::shared_ptr<MixedNvqDispatch>& dispatch);
    MoeActivationWorkspace activation_workspace(
            ::tyloi::DenseTensor x, int input_rows, int groups, int gs,
            MixedMoeTransformKey transform, bool stable = true);
    MoeActivationWorkspace prefill_activation_workspace(
            ::tyloi::DenseTensor x, int input_rows, int groups, int gs,
            MixedMoeTransformKey transform);
};

namespace tyloi::cuda {
struct MoeScratchCache {
    std::map<std::weak_ptr<const void>,
             std::map<const void*, std::map<std::pair<int, std::uintptr_t>, std::shared_ptr<MoeScratch>>>,
             std::owner_less<std::weak_ptr<const void>>> values;
    mutable std::mutex mutex;

    std::shared_ptr<MoeScratch> get(const std::shared_ptr<const void>& source, ::tyloi::DenseTensor input);
};
}

struct MixedMoeRuntime {
    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    bool partial_experts = false;
    // A streamed prompt group is still prefill when a rare expert has <=8
    // rows. Its workspaces are shared by capacity, not cached per row count.
    bool prefill_only = false;
    std::vector<MixedMoePool> pools;
    std::shared_ptr<MixedNvqDispatch> nvq_dispatch;
    std::shared_ptr<MixedNintDispatch> nint_dispatch;
    bool nint_only() const;

    ::tyloi::DenseTensor forward(
            MoeScratch& scratch,
            const MoeOptions& options,
            KlMmqState& kl_mmq,
            ::tyloi::DenseTensor x,
            const MoeRoutePlan & route,
            bool input_prequantized = false,
            int epilogue_mode = 0) const;

    ::tyloi::DenseTensor forward_glu_output(
            MoeScratch& scratch,
            const MoeOptions& options,
            KlMmqState& kl_mmq,
            ::tyloi::DenseTensor x,
            const MoeRoutePlan & route,
            bool gelu) const;

    bool supports_clamped_swiglu() const;

    ::tyloi::DenseTensor forward_clamped_swiglu(
            MoeScratch& scratch,
            const MoeOptions& options,
            KlMmqState& kl_mmq,
            ::tyloi::DenseTensor gate_up,
            const MoeRoutePlan & route,
            double limit) const;
};
