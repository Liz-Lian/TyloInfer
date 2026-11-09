#pragma once

#include "engine/backends/cuda/ops/ffn/options.h"
#include "engine/ops.h"

#include <memory>
#include <optional>
#include <vector>

struct MoeRoutePlan;

namespace tyloi::cuda {

class Executor;
class Projector;

class Ffn {
public:
    std::vector<std::shared_ptr<const ::tyloi::Tensor>> gate_up;
    std::shared_ptr<const ::tyloi::Tensor> down;
    std::unique_ptr<Ffn> important_neurons;
    bool geglu = false;
    // Dense Ffn and shared/routed experts share the explicit activation label.
    tyloi::ops::MlpActivation activation = tyloi::ops::MlpActivation::silu_gated;
    bool is_moe = false;
    bool moe_split_gate_up = false;
    // Routed experts without a gate projection use the activation above.
    bool moe_ungated_experts = false;
    std::shared_ptr<const ::tyloi::Tensor> moe_gate_up;
    std::shared_ptr<const ::tyloi::Tensor> moe_gate;
    std::shared_ptr<const ::tyloi::Tensor> moe_up;
    std::shared_ptr<const ::tyloi::Tensor> moe_down;
    // Optional latent projections around the routed branch only.
    std::shared_ptr<const ::tyloi::Tensor> moe_latent_in;
    std::shared_ptr<const ::tyloi::Tensor> moe_latent_out;
    std::shared_ptr<const ::tyloi::Tensor> moe_router;
    std::shared_ptr<const ::tyloi::Tensor> moe_shared_gate;
    std::shared_ptr<const ::tyloi::Tensor> moe_router_bias;
    std::shared_ptr<const ::tyloi::Tensor> moe_hash_ids;
    std::unique_ptr<Ffn> shared;
    int moe_top_k = 0;
    bool moe_use_sigmoid = false;
    bool moe_use_sqrt_softplus = false;
    bool moe_normalize = false;
    bool moe_delayed_softmax = true;
    bool moe_shared_ungated = false;
    double moe_router_scale = 1.0;
    double swiglu_limit = 0.0;
    int moe_layer = -1;

#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::DenseTensor forward(
        ::tyloi::DenseTensor input,
        std::optional<::tyloi::DenseTensor> input_ids = std::nullopt,
        std::optional<::tyloi::DenseTensor> router_bias = std::nullopt) const;
#else
    ::tyloi::DenseTensor forward(
        ::tyloi::DenseTensor input, Executor& executor, const FfnMode& mode,
        std::optional<::tyloi::DenseTensor> input_ids = std::nullopt,
        std::optional<::tyloi::DenseTensor> router_bias = std::nullopt) const;
#endif

private:
    ::tyloi::DenseTensor forward_impl(
        ::tyloi::DenseTensor input, Executor& executor, const FfnMode& mode,
        const Projector& project, std::optional<::tyloi::DenseTensor> input_ids,
        bool allow_important_neurons,
        std::optional<::tyloi::DenseTensor> router_bias) const;
    ::tyloi::DenseTensor forward_experts(
        ::tyloi::DenseTensor input, Executor& executor, const FfnMode& mode,
        const Projector& project, const MoeRoutePlan& route,
        ::tyloi::DenseTensor route_weights) const;
};

} // namespace tyloi::cuda
