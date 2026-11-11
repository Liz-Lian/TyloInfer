#pragma once

#include "config.h"
#include "../common/check.h"
#include <array>
#include <cmath>

namespace tyloi::models::gemma4 {

struct LayerSpec {
    bool moe, sliding, value_equals_key;
    int64_t heads, kv_heads, head_dim, window, rotary_pairs;
    double rope_base;
};

inline LayerSpec layer_spec(const Config &c, std::string_view type) {
    require_model(type == "full_attention" || type == "sliding_attention",
                  "unsupported Gemma4 layer type");
    const bool sliding = type == "sliding_attention";
    const auto width = sliding ? c.head_dim : c.global_head_dim;
    return {c.num_experts > 0,
            sliding,
            !sliding && c.attention_key_equals_value,
            c.num_attention_heads,
            sliding ? c.num_key_value_heads : c.num_global_key_value_heads,
            width,
            sliding ? c.sliding_window : 0,
            sliding ? -1 : int64_t(std::llround(c.full_rotary_factor * double(width) / 2.0)),
            sliding ? c.sliding_rope_base : c.rope_base};
}

template <class Attention, class Layer, class Loader>
void load_layer(Attention &a, Layer &b, Loader &ops, const Config &c, int layer, std::string_view type) {
    const auto spec = layer_spec(c, type);
    a.layer = layer;
    b.gemma4_moe = spec.moe;
    a.max_position_embeddings = c.max_position_embeddings;
    a.rms_norm_eps = c.rms_norm_eps;
    a.norm_weight_offset = 0.0;
    a.sliding = spec.sliding;
    a.value_equals_key = spec.value_equals_key;
    a.attention_heads = spec.heads;
    a.kv_heads = spec.kv_heads;
    a.attention_head_dim = a.attention_rotary_dim = spec.head_dim;
    a.attention_window = spec.window;
    a.attention_scale = 1.0;
    a.attention_rope = ops.rope(c.max_position_embeddings, spec);
    const auto root = "model.block." + std::to_string(layer) + ".";
    const auto attention = root + "attention.";
    a.attn_norm = ops.dense(attention + "norm.weight");
    b.attn_post_norm = ops.dense(attention + "output_norm.weight");
    a.q_norm = ops.dense(attention + "query_norm.weight");
    a.k_norm = ops.dense(attention + "key_norm.weight");
    a.v_norm = ops.ones(spec.head_dim);
    std::vector<std::string> projections{attention + "query.weight", attention + "key.weight"};
    if (!spec.value_equals_key)
        projections.push_back(attention + "value.weight");
    ops.qkv(a, projections);
    ops.output(a, attention + "output.weight");
    const auto mlp = root + "mlp.";
    b.ffn_norm = ops.dense(mlp + "dense.input_norm.weight");
    b.ffn_post_norm = ops.dense(mlp + "output_norm.weight");
    b.layer_scale = ops.layer_scale(root + "output_scale");
    if (spec.moe) {
        b.ffn_post_norm_1 = ops.dense(mlp + "dense.output_norm.weight");
        b.ffn_pre_norm_2 = ops.dense(mlp + "experts.input_norm.weight");
        b.ffn_post_norm_2 = ops.dense(mlp + "experts.output_norm.weight");
    }
    b.ffn = ops.mlp({mlp + "gate.weight", mlp + "up.weight"}, mlp + "down.weight");
    if (!spec.moe)
        return;
    b.gemma_moe_gate_up = ops.experts(mlp + "experts.gate_up.weight", layer, "gate_up");
    b.gemma_moe_down = ops.experts(mlp + "experts.down.weight", layer, "down");
    b.gemma_router = ops.router_parameter(mlp + "router.weight");
    b.gemma_router_norm_scale = ops.normalized_router_parameter(mlp + "router.norm.weight",
                                                                std::sqrt(double(c.hidden_size)));
    b.gemma_expert_scale = ops.router_parameter(mlp + "router.expert_scale");
    b.gemma_top_k = static_cast<int>(c.num_experts_per_tok);
    require_model(
        ops.expert_shape(b.gemma_moe_gate_up) ==
                std::array<int64_t, 3>{c.num_experts, 2 * c.moe_intermediate_size, c.hidden_size} &&
            ops.expert_shape(b.gemma_moe_down) ==
                std::array<int64_t, 3>{c.num_experts, c.hidden_size, c.moe_intermediate_size} &&
            ops.shape(b.gemma_router) == std::vector<int64_t>{c.num_experts, c.hidden_size} &&
            ops.elements(b.gemma_router_norm_scale) == c.hidden_size &&
            ops.elements(b.gemma_expert_scale) == c.num_experts,
        "Gemma4 MoE tensor shapes disagree with config");
}

} // namespace tyloi::models::gemma4
