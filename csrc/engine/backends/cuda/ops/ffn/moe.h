#pragma once

#include "engine/weights/encoding/mfe.h"
#include "engine/weights/moe/dispatch.h"
#include "engine/backends/cuda/ops/ffn/activation.h"
#include "engine/backends/cuda/ops/ffn/routes.h"
#include "engine/backends/cuda/ops/projection/mmq.h"

#include <cstdint>
#include <array>
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
class Projector;
class MoeSharedExpert {
public:
    MoeSharedExpert(::tyloi::Tensor gate, ::tyloi::Tensor up, ::tyloi::Tensor down,
                    ::tyloi::Tensor scalar_gate);
    ::tyloi::DenseTensor forward(const Projector&, const ::tyloi::DenseTensor&, std::int64_t rows) const;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::DenseTensor forward(const ::tyloi::DenseTensor&, std::int64_t rows) const;
#endif
    ::tyloi::DenseTensor gate(const Projector&, const ::tyloi::DenseTensor&) const;
    std::array<::tyloi::Tensor,3> projections;
    ::tyloi::Tensor scalar_gate;
private:
    ::tyloi::DenseTensor raw(const Projector&, const ::tyloi::DenseTensor&) const;
    bool tq = false, row_parallel = false;
};

struct MoeScratchCache {
    std::map<std::weak_ptr<const void>,
             std::map<const void*, std::map<std::pair<int, std::uintptr_t>, std::shared_ptr<MoeScratch>>>,
             std::owner_less<std::weak_ptr<const void>>> values;
    mutable std::mutex mutex;

    std::shared_ptr<MoeScratch> get(const std::shared_ptr<const void>& source, ::tyloi::DenseTensor input);
};
}

namespace tyloi::cuda {
DenseTensor mixed_project(const MfeTensor& weight, MoeScratch& scratch,
    const MoeOptions& options, KlMmqState& kl_mmq, DenseTensor x,
    const MoeRoutePlan& route, bool input_prequantized = false, int epilogue_mode = 0);
DenseTensor mixed_glu(const MfeTensor& weight, MoeScratch& scratch,
    const MoeOptions& options, KlMmqState& kl_mmq, DenseTensor x,
    const MoeRoutePlan& route, bool gelu);
bool supports_clamp(const MfeTensor& weight);
DenseTensor mixed_clamp(const MfeTensor& weight, MoeScratch& scratch,
    const MoeOptions& options, KlMmqState& kl_mmq, DenseTensor gate_up,
    const MoeRoutePlan& route, double limit);
}
