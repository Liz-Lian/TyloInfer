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
    std::vector<::tyloi::Tensor> gate_up;
    ::tyloi::Tensor down;
    std::unique_ptr<Ffn> important_neurons;
    bool geglu = false;
    // Dense Ffn and shared/routed experts share the explicit activation label.
    tyloi::ops::MlpActivation activation = tyloi::ops::MlpActivation::silu_gated;
    bool is_moe = false;
    bool moe_split_gate_up = false;
    // Routed experts without a gate projection use the activation above.
    bool moe_ungated_experts = false;
    ::tyloi::Tensor moe_gate_up;
    ::tyloi::Tensor moe_gate;
    ::tyloi::Tensor moe_up;
    ::tyloi::Tensor moe_down;
    // Optional latent projections around the routed branch only.
    ::tyloi::Tensor moe_latent_in;
    ::tyloi::Tensor moe_latent_out;
    ::tyloi::Tensor moe_router;
    ::tyloi::Tensor moe_shared_gate;
    ::tyloi::Tensor moe_router_bias;
    ::tyloi::Tensor moe_hash_ids;
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

};

} // namespace tyloi::cuda
