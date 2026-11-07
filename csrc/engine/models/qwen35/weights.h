#pragma once

#include "engine/models/qwen35/config.h"
#include "engine/ops.h"
#include "../common/weight_loading.h"

namespace tyloi::models::qwen35 {

struct LinearWeightNames {
    std::string qkv, qk, value, gate, alpha, beta;
};

template <class Ffn, class Loader>
Ffn load_ffn(Loader& ops, const Config& config, int layer, std::string_view tensor_root) {
    const std::string prefix =
        std::string(tensor_root) + ".block." + std::to_string(layer) + ".mlp.";
    if (!ops.has(prefix + "experts.gate_up.weight") &&
        !ops.has(prefix + "experts.gate.weight") &&
        !ops.has(prefix + "experts.up.weight") &&
        !ops.has(prefix + "experts.down.weight")) {
        return models::load_dense_ffn<Ffn>(ops, config, prefix);
    }
    if (config.num_experts <= 0 || config.num_experts_per_tok <= 0 ||
        config.moe_intermediate_size <= 0 || config.shared_expert_intermediate_size <= 0) {
        throw std::runtime_error("Qwen MoE config fields are missing");
    }

    auto ffn = models::load_moe_weights<Ffn>(ops, prefix, layer);
    ffn.moe_shared_gate = ops.router_parameter(prefix + "shared_expert.router.weight");
    ffn.moe_top_k = static_cast<int>(config.num_experts_per_tok);
    const auto routing = config.routing();
    ffn.moe_use_sqrt_softplus = routing.activation == tyloi::ops::RouterActivation::sqrt_softplus;
    ffn.moe_normalize = routing.normalize;
    ffn.moe_delayed_softmax = routing.delayed_softmax;
    ffn.moe_router_scale = routing.scale;
    const bool routed_gate_shapes =
        ffn.moe_split_gate_up
            ? ffn.moe_gate && ffn.moe_up && ffn.moe_gate->size(0) == config.num_experts &&
                  ffn.moe_up->size(0) == config.num_experts &&
                  ffn.moe_gate->size(2) == config.hidden_size &&
                  ffn.moe_up->size(2) == config.hidden_size &&
                  ffn.moe_gate->size(1) == config.moe_intermediate_size &&
                  ffn.moe_up->size(1) == config.moe_intermediate_size
            : ffn.moe_gate_up && ffn.moe_gate_up->size(0) == config.num_experts &&
                  ffn.moe_gate_up->size(2) == config.hidden_size &&
                  ffn.moe_gate_up->size(1) == 2 * config.moe_intermediate_size;
    if (!routed_gate_shapes || !ffn.moe_down || ffn.moe_down->size(0) != config.num_experts ||
        ffn.moe_down->size(2) != config.moe_intermediate_size ||
        ffn.moe_down->size(1) != config.hidden_size || ops.shape(ffn.moe_router).size() != 2 ||
        ops.shape(ffn.moe_router)[0] != config.num_experts ||
        ops.shape(ffn.moe_router)[1] != config.hidden_size || ops.shape(ffn.moe_shared_gate).size() != 2 ||
        ops.shape(ffn.moe_shared_gate)[0] != 1 || ops.shape(ffn.moe_shared_gate)[1] != config.hidden_size) {
        throw std::runtime_error("Qwen MoE tensor shapes disagree with config at layer " +
                                 std::to_string(layer));
    }
    return ffn;
}

} // namespace tyloi::models::qwen35
