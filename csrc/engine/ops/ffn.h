#pragma once
#include "engine/ops.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace tyloi::ops {
/** Select the numerical scheduling used by an FFN's routed branch. */
enum class FfnPolicy { standard, qwen4 };
/** Immutable tensor weights and numerical policy for dense or routed FFNs. */
struct FfnWeights {
    std::vector<Tensor> gate_up;
    Tensor down;
    std::unique_ptr<FfnWeights> important_neurons;
    bool geglu = false;
    MlpActivation activation = MlpActivation::silu_gated;
    bool is_moe = false, moe_split_gate_up = false, moe_ungated_experts = false;
    Tensor moe_gate_up, moe_gate, moe_up, moe_down, moe_latent_in, moe_latent_out;
    Tensor moe_router, moe_shared_gate, moe_router_bias, moe_hash_ids;
    std::unique_ptr<FfnWeights> shared;
    int moe_top_k = 0;
    bool moe_use_sigmoid = false, moe_use_sqrt_softplus = false, moe_normalize = false;
    bool moe_delayed_softmax = true, moe_shared_ungated = false;
    double moe_router_scale = 1.0, swiglu_limit = 0.0;
    int moe_layer = -1;
    FfnPolicy policy = FfnPolicy::standard;
    /** CUDA execution device used when the input is on the host. */
    Device target;
    std::int64_t qwen_hidden = 0, qwen_moe_width = 0;
    bool qwen_tq = false;
};
/** Validate Qwen4 routed banks and finalize their execution metadata. */
FfnWeights qwen4_ffn_weights(FfnWeights weights);
/** Execute the FFN using active backend scheduling and optional routing inputs. */
Tensor ffn(const FfnWeights& weights, const Tensor& input, const Tensor& ids = {},
           const Tensor& router_bias = {}, std::int64_t projection_rows = 0);
} // namespace tyloi::ops
