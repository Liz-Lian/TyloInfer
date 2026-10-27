#pragma once

#include "config.h"
#include "../common/weight_loading.h"

namespace mfq::models::qwen35 {

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
    ffn.moe_use_sqrt_softplus = routing.activation == mfq::models::RouterActivation::sqrt_softplus;
    ffn.moe_normalize = routing.normalize;
    ffn.moe_delayed_softmax = routing.delayed_softmax;
    ffn.moe_router_scale = routing.scale;
    const bool routed_gate_shapes =
        ffn.moe_split_gate_up
            ? ffn.moe_gate.n_experts == config.num_experts &&
                  ffn.moe_up.n_experts == config.num_experts &&
                  ffn.moe_gate.neuron_len == config.hidden_size &&
                  ffn.moe_up.neuron_len == config.hidden_size &&
                  ffn.moe_gate.out_per_expert == config.moe_intermediate_size &&
                  ffn.moe_up.out_per_expert == config.moe_intermediate_size
            : ffn.moe_gate_up.n_experts == config.num_experts &&
                  ffn.moe_gate_up.neuron_len == config.hidden_size &&
                  ffn.moe_gate_up.out_per_expert == 2 * config.moe_intermediate_size;
    if (!routed_gate_shapes || ffn.moe_down.n_experts != config.num_experts ||
        ffn.moe_down.neuron_len != config.moe_intermediate_size ||
        ffn.moe_down.out_per_expert != config.hidden_size || ops.shape(ffn.moe_router).size() != 2 ||
        ops.shape(ffn.moe_router)[0] != config.num_experts ||
        ops.shape(ffn.moe_router)[1] != config.hidden_size || ops.shape(ffn.moe_shared_gate).size() != 2 ||
        ops.shape(ffn.moe_shared_gate)[0] != 1 || ops.shape(ffn.moe_shared_gate)[1] != config.hidden_size) {
        throw std::runtime_error("Qwen MoE tensor shapes disagree with config at layer " +
                                 std::to_string(layer));
    }
    return ffn;
}

template <class Attention, class Loader>
void load_linear_attention(Attention& attention, Loader& ops, const Config& config,
                           int layer, std::string_view tensor_root = "model") {
    const auto prefix = std::string(tensor_root) + ".block." + std::to_string(layer) + ".";
    const auto linear = prefix + "linear_attention.";
    LinearWeightNames names{linear + "qkv.weight", linear + "qk.weight", linear + "value.weight",
                            linear + "gate.weight", linear + "alpha.weight", linear + "beta.weight"};
    ops.linear_projections(attention, names, ops.has(names.qk) && ops.has(names.value));
    attention.conv_weight = ops.dense(linear + "conv.weight");
    if (ops.has(linear + "conv.bias")) attention.conv_bias = ops.dense(linear + "conv.bias");
    attention.dt_bias = ops.dense(linear + "dt_bias");
    auto a = ops.dense(linear + "a");
    attention.a_log = config.legacy_tensor_layout.linear_attention_a_is_log
        ? a : ops.log_negative(a);
    attention.linear_norm = ops.dense(linear + "norm.weight");
    ops.linear_output(attention, linear + "output.weight");
}


template <class Loader>
auto load_block(Loader& ops, const Config& config, int layer, const std::string& type,
                std::string_view tensor_root = "model") {
    const auto prefix = std::string(tensor_root) + ".block." + std::to_string(layer) + ".";
    if (Config::attention_kind(type) == Config::AttentionKind::full) {
        auto block = ops.full_block();
        models::load_full_attention(*block, ops, config, layer, prefix,
                                    config.legacy_tensor_layout.norm_weight_offset,
                                    config.attention_output_gate);
        block->ffn = load_ffn<typename Loader::Ffn>(ops, config, layer, tensor_root);
        return ops.finish(std::move(block));
    }
    auto block = ops.linear_block();
    block->qwen_config = config;
    block->tiled_v_heads = config.legacy_tensor_layout.qwen_gdn_gguf_layout;
    block->attn_norm = ops.dense(prefix + "attention.norm.weight");
    block->ffn_norm = ops.dense(prefix + "mlp.norm.weight");
    load_linear_attention(*block, ops, config, layer, tensor_root);
    block->ffn = load_ffn<typename Loader::Ffn>(ops, config, layer, tensor_root);
    return ops.finish(std::move(block));
}

} // namespace mfq::models::qwen35
