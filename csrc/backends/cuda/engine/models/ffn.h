#pragma once

#include "ops/common/include/quant_linear.h"
#include "mfe_weight.h"
#include "engine/models/common/gated_mlp.h"

#include <memory>

class MoeFfnPipeline;
struct TqMoeBanks;

namespace mfq::cuda {

struct Ffn {
    QuantLinearGroup gate_up;
    QuantLinear down;
    std::unique_ptr<Ffn> important_neurons;
    bool geglu = false;
    // Dense Ffn and shared/routed experts share the explicit activation label.
    mfq::models::MlpActivation activation = mfq::models::MlpActivation::silu_gated;
    bool is_moe = false;
    bool moe_split_gate_up = false;
    // Routed experts without a gate projection use the activation above.
    bool moe_ungated_experts = false;
    MfeWeight moe_gate_up;
    MfeWeight moe_gate;
    MfeWeight moe_up;
    MfeWeight moe_down;
    // Ordinary TQ row tables retain their own cache and never enter MFE pools.
    std::shared_ptr<TqMoeBanks> moe_tq;
    std::shared_ptr<MixedMoeRuntime> cpu_moe_gate_up;
    std::shared_ptr<MixedMoeRuntime> cpu_moe_gate;
    std::shared_ptr<MixedMoeRuntime> cpu_moe_up;
    std::shared_ptr<MixedMoeRuntime> cpu_moe_down;
    // Dense (F32/F16/BF16) routed expert banks: [experts,out,in].
    mfq_tensor_backend::Tensor moe_dense_gate;
    mfq_tensor_backend::Tensor moe_dense_up;
    mfq_tensor_backend::Tensor moe_dense_down;
    // Optional latent projections around the routed branch only.
    QuantLinear moe_latent_in;
    QuantLinear moe_latent_out;
    mfq_tensor_backend::Tensor moe_router;
    std::shared_ptr<const QuantLinear> moe_router_projection;
    mfq_tensor_backend::Tensor moe_shared_gate;
    mfq_tensor_backend::Tensor moe_router_bias;
    mfq_tensor_backend::Tensor moe_hash_ids;
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
    std::weak_ptr<MoeFfnPipeline> moe_pipeline;
};

} // namespace mfq::cuda
