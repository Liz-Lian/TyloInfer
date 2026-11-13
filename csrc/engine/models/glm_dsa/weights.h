#pragma once
#include "config.h"
#include "engine/models/common/load.h"
#include "engine/models/metadata.h"
#include <utility>

namespace tyloi::models::glm_dsa {

inline void load_config(std::string_view definition, Config& c, CausalLmMetadata& m) {
    c = Config::from_json(definition);
    c.layer_types.assign(static_cast<size_t>(c.num_hidden_layers), "glm_dsa");
    c.rotary_dim = c.qk_rope_head_dim;
    m.vocab_size = c.vocab_size; m.hidden_size = c.hidden_size;
    m.num_hidden_layers = c.num_hidden_layers; m.num_attention_heads = c.num_attention_heads;
    m.num_key_value_heads = c.num_key_value_heads; m.head_dim = c.head_dim;
    m.max_position_embeddings = c.max_position_embeddings; m.rotary_dim = c.rotary_dim;
    m.num_experts = c.num_experts; m.rope_base = c.rope_base; m.rms_norm_eps = c.rms_norm_eps;
    m.norm_weight_offset = 1.0; m.tie_word_embeddings = c.tie_word_embeddings;
    m.decode_graph_double_warmup = true; m.model_type = c.model_type; m.layer_types = c.layer_types;
}

template <class Ffn, class Loader>
void load_ffn(Loader& ops, const Config& c, int i, Ffn& f) {
    const auto &config = c;
    const std::string p = "model.block." + std::to_string(i) + ".mlp.";
    if (config.mlp_layer_types.at(static_cast<size_t>(i)) == "sparse") {
        models::load_moe_weights_into(f, ops, p, i, true);
        f.moe_top_k = static_cast<int>(c.num_experts_per_tok);
        f.moe_use_sigmoid = true;
        f.moe_use_sqrt_softplus = false;
        f.moe_normalize = c.norm_topk_prob;
        f.moe_delayed_softmax = false;
        f.moe_shared_ungated = true;
        f.moe_router_scale = c.routed_scaling_factor;
        if (!f.moe_gate_up || !f.moe_down ||
            f.moe_gate_up.size(0) != c.num_experts || f.moe_down.size(0) != c.num_experts ||
            f.moe_gate_up.size(2) != c.hidden_size ||
            f.moe_gate_up.size(1) != 2 * c.moe_intermediate_size ||
            f.moe_down.size(2) != c.moe_intermediate_size ||
            f.moe_down.size(1) != c.hidden_size || ops.shape(f.moe_router).size() != 2 ||
            ops.shape(f.moe_router)[0] != c.num_experts || ops.shape(f.moe_router)[1] != c.hidden_size ||
            ops.elements(f.moe_router_bias) != c.num_experts) {
            throw std::runtime_error("GLM DSA MoE tensor shapes disagree with config at layer " +
                                     std::to_string(i));
        }
        return;
    }
    f = models::load_dense_ffn<Ffn>(ops, c, p);
}

template <class Block, class Loader>
void load_layer_weights(Block& b, Loader& ops, const Config& c, int i, const std::string& type) {
    const auto &config = c;
    if (type != "glm_dsa") {
        throw std::runtime_error("invalid GLM DSA block loader state");
    }
    const std::string lp = "model.block." + std::to_string(i) + ".";
    const std::string ap = lp + "attention.";
    b.config = c;
    b.layer = i;
    b.full_indexer = config.indexer_types.at(static_cast<size_t>(i)) == "full";
    b.attn_norm = ops.weights.tensor(ap + "norm.weight");
    b.ffn_norm = ops.weights.tensor(lp + "mlp.norm.weight");
    b.q_a_norm = ops.weights.tensor(ap + "query_a_norm.weight");
    b.kv_a_norm = ops.weights.tensor(ap + "key_value_a_norm.weight");
    std::vector<std::string> first_names = {
        ap + "query_a.weight",
        ap + "key_value_a.weight",
    };
    std::vector<std::string> second_names = {
        ap + "query_b.weight",
    };
    if (b.full_indexer) {
        first_names.push_back(ap + "indexer.key.weight");
        first_names.push_back(ap + "indexer.score.weight");
        second_names.push_back(ap + "indexer.query.weight");
        b.index_k_norm = ops.weights.tensor(ap + "indexer.key_norm.weight");
        b.index_k_bias = ops.weights.tensor(ap + "indexer.key_norm.bias");
    }
    b.input_proj = ops.projections(first_names);
    b.q_proj = ops.projections(second_names);
    b.embed_q = ops.headwise(ap + "latent.query_embedding.weight");
    b.unembed_out = ops.headwise(ap + "latent.output_unembedding.weight");
    b.o_proj = ops.linear(ap + "output.weight");
    load_ffn(ops, c, i, b.ffn);

    const bool input_shape_ok = b.input_proj.size() == (b.full_indexer ? 4u : 2u) &&
                                b.input_proj[0].size(0) == c.q_lora_rank &&
                                b.input_proj[1].size(0) == c.kv_lora_rank + c.qk_rope_head_dim &&
                                (!b.full_indexer || (b.input_proj[2].size(0) == c.index_head_dim &&
                                                      b.input_proj[3].size(0) == c.index_n_heads));
    const bool q_shape_ok =
        b.q_proj.size() == (b.full_indexer ? 2u : 1u) &&
        b.q_proj[0].size(0) == c.num_attention_heads * (c.qk_nope_head_dim + c.qk_rope_head_dim) &&
        (!b.full_indexer || b.q_proj[1].size(0) == c.index_n_heads * c.index_head_dim);
    const bool head_shape_ok = b.embed_q.size(0) == c.num_attention_heads &&
                               b.embed_q.size(2) == c.qk_nope_head_dim &&
                               b.embed_q.size(1) == c.kv_lora_rank &&
                               b.unembed_out.size(0) == c.num_attention_heads &&
                               b.unembed_out.size(2) == c.kv_lora_rank &&
                               b.unembed_out.size(1) == c.v_head_dim &&
                               b.o_proj.size(1) == c.num_attention_heads * c.v_head_dim &&
                               b.o_proj.size(0) == c.hidden_size;
    if (!input_shape_ok || !q_shape_ok || !head_shape_ok || ops.elements(b.attn_norm) != c.hidden_size ||
        ops.elements(b.ffn_norm) != c.hidden_size || ops.elements(b.q_a_norm) != c.q_lora_rank ||
        ops.elements(b.kv_a_norm) != c.kv_lora_rank ||
        (b.full_indexer && (ops.elements(b.index_k_norm) != c.index_head_dim ||
                             ops.elements(b.index_k_bias) != c.index_head_dim))) {
        throw std::runtime_error("GLM DSA tensor shapes disagree with config at layer " +
                                 std::to_string(i));
    }
}


} // namespace tyloi::models::glm_dsa
