#pragma once
#include "config.h"
#include "engine/models/metadata.h"
#include "../common/check.h"
#include "../common/recurrent_attention.h"
#include <memory>

namespace tyloi::models::glm5_next {

template <class Loader>
auto load_dense_ffn(Loader &ops, const std::string &p, double limit) {
    auto gate = ops.weight(p + ".gate.weight"), up = ops.weight(p + ".up.weight"),
         down = ops.weight(p + ".down.weight");
    return ops.gated_mlp(std::move(gate), std::move(up), std::move(down), limit);
}

template <class Loader>
auto load_ffn(Loader &ops, const Config &c, int layer, const std::string &p, bool predictor) {
    if (c.dense_layer(layer, predictor))
        return load_dense_ffn(ops, p, c.swiglu_limit);
    auto gate_up = ops.routed_gate_up(p, layer, c.experts, c.moe_intermediate, c.hidden);
    auto down = ops.routed(p + ".experts.down.weight", layer, c.experts, c.hidden, c.moe_intermediate);
    auto router = ops.weight(p + ".router.weight");
    auto shared = load_dense_ffn(ops, p + ".shared_expert", c.swiglu_limit);
#ifdef TYLOINFER_NATIVE_CUDA
    auto bias = std::make_shared<typename Loader::DenseTensor>(
        ops.dense(p + ".router.bias").dense()
            .to(::tyloi::kFloat32).contiguous());
#else
    auto bias = ops.dense(p + ".router.bias").dense()
                    .to(::tyloi::kFloat32).contiguous();
#endif
    return ops.moe(std::move(gate_up), std::move(down), std::move(router), std::move(shared),
                   std::move(bias), c);
}

template <class Loader>
auto load_mla(Loader &ops, const Config &c, int layer, const std::string &a) {
    typename Loader::MlaWeights w{
        ops.weight(a + ".query_a.weight"), ops.weight(a + ".key_value_a.weight"),
        ops.weight(a + ".query_b.weight"), ops.weight(a + ".output.weight"),
        ops.weight(a + ".indexer.query.weight"), ops.weight(a + ".indexer.key.weight"),
        ops.weight(a + ".indexer.score.weight"),
        ops.routed(a + ".latent.query_embedding.weight", layer, c.heads, c.latent, c.nope),
        ops.routed(a + ".latent.output_unembedding.weight", layer, c.heads, c.value_width, c.latent),
#ifdef TYLOINFER_NATIVE_CUDA
        ops.dense(a + ".query_a_norm.weight"),
        ops.dense(a + ".key_value_a_norm.weight"),
        ops.dense(a + ".indexer.key_norm.weight"),
        ops.dense(a + ".indexer.key_norm.bias"),
        ops.dense(a + ".indexer.pool.gate"),
        ops.dense(a + ".indexer.pool.position")};
#else
        ops.dense(a + ".query_a_norm.weight").dense(),
        ops.dense(a + ".key_value_a_norm.weight").dense(),
        ops.dense(a + ".indexer.key_norm.weight").dense(),
        ops.dense(a + ".indexer.key_norm.bias").dense(),
        ops.dense(a + ".indexer.pool.gate").dense(),
        ops.dense(a + ".indexer.pool.position").dense()};
#endif
    return ops.mla(std::move(w), c);
}

template <class Loader>
auto load_mhc(Loader &ops, const std::string &p) {
#ifdef TYLOINFER_NATIVE_CUDA
    auto function = ops.dense(p + ".function");
    auto base = ops.dense(p + ".base");
    auto scale = ops.dense(p + ".scale");
#else
    auto function = ops.dense(p + ".function").dense();
    auto base = ops.dense(p + ".base").dense();
    auto scale = ops.dense(p + ".scale").dense();
#endif
    return ops.mhc(std::move(function), std::move(base), std::move(scale));
}

template <class Layer, class Loader>
void load_layer_weights(Layer &b, Loader &ops, const Config &c, int layer) {
    const auto p = "model.block." + std::to_string(layer);
    b.config = c;
    b.attention_hc = load_mhc(ops, p + ".attention.mhc.pre");
    b.ffn_hc = load_mhc(ops, p + ".mlp.mhc.pre");
#ifdef TYLOINFER_NATIVE_CUDA
    b.attention_norm = ops.dense(p + ".attention.norm.weight");
    b.ffn_norm = ops.dense(p + ".mlp.norm.weight");
#else
    b.attention_norm = ops.dense(p + ".attention.norm.weight").dense();
    b.ffn_norm = ops.dense(p + ".mlp.norm.weight").dense();
#endif
    b.ffn = load_ffn(ops, c, layer, p + ".mlp", false);
    if (c.linear_layer(layer)) {
        const auto a = p + ".linear_attention";
        typename Loader::KdaWeights w{
            ops.weight(a + ".query.weight"), ops.weight(a + ".key.weight"),
            ops.weight(a + ".value.weight"), ops.weight(a + ".beta.weight"),
            ops.weight(a + ".gate_a.weight"), ops.weight(a + ".gate_b.weight"),
            ops.weight(a + ".output.weight"),
#ifdef TYLOINFER_NATIVE_CUDA
            ops.concat({ops.dense(a + ".query_conv.weight"),
                        ops.dense(a + ".key_conv.weight"),
                        ops.dense(a + ".value_conv.weight")}),
            ops.dense(a + ".forget_a.weight"),
            ops.dense(a + ".forget_b.weight"),
            ops.dense(a + ".dt_bias"),
            ops.dense(a + ".a"),
            ops.dense(a + ".output_norm.weight")};
#else
            ops.concat({ops.dense(a + ".query_conv.weight").dense(),
                        ops.dense(a + ".key_conv.weight").dense(),
                        ops.dense(a + ".value_conv.weight").dense()}),
            ops.dense(a + ".forget_a.weight").dense(),
            ops.dense(a + ".forget_b.weight").dense(),
            ops.dense(a + ".dt_bias").dense(),
            ops.dense(a + ".a").dense(),
            ops.dense(a + ".output_norm.weight").dense()};
#endif
        b.kda = ops.kda(std::move(w), c);
    } else {
        b.mla = load_mla(ops, c, layer, p + ".attention");
    }
}

template <class Predictor, class Loader>
bool load_predictor(Predictor &result, Loader &ops, const Config &c) {
    const auto count = c.predictor_layers;
    if (!ops.has("predictor.embedding_norm.weight") || count <= 0) {
        require_model(!ops.has_prefix("predictor"),
                      "GLM5-Next model source contains an incomplete or undeclared MTP head");
        return false;
    }
    result.config = c;
#ifdef TYLOINFER_NATIVE_CUDA
    result.embedding_norm = std::make_shared<typename Loader::DenseTensor>(
        ops.dense("predictor.embedding_norm.weight").dense()
            .to(::tyloi::kFloat32).contiguous());
    result.hidden_norm = std::make_shared<typename Loader::DenseTensor>(
        ops.dense("predictor.hidden_norm.weight").dense()
            .to(::tyloi::kFloat32).contiguous());
    result.output_norm = ops.dense("predictor.output_norm.weight");
#else
    result.embedding_norm = ops.dense("predictor.embedding_norm.weight").dense().to(::tyloi::kFloat32).contiguous();
    result.hidden_norm = ops.dense("predictor.hidden_norm.weight").dense().to(::tyloi::kFloat32).contiguous();
    result.output_norm = ops.dense("predictor.output_norm.weight").dense();
#endif
    result.fusion = ops.weight("predictor.fusion.weight");
    for (int64_t i = 0; i < count; ++i) {
        const auto p = "predictor.block." + std::to_string(i);
        typename Predictor::Layer layer;
#ifdef TYLOINFER_NATIVE_CUDA
        layer.attention_norm = ops.dense(p + ".attention.norm.weight");
        layer.ffn_norm = ops.dense(p + ".mlp.norm.weight");
#else
        layer.attention_norm = ops.dense(p + ".attention.norm.weight").dense();
        layer.ffn_norm = ops.dense(p + ".mlp.norm.weight").dense();
#endif
        layer.ffn = load_ffn(ops, c, int(i), p + ".mlp", true);
        layer.attention = load_mla(ops, c, int(i), p + ".attention");
        result.layers.push_back(std::move(layer));
    }
#ifdef TYLOINFER_NATIVE_CUDA
    require_model(result.embedding_norm.dim() == 1 && result.embedding_norm.size(0) == c.hidden &&
                      result.hidden_norm.dim() == 1 && result.hidden_norm.size(0) == c.hidden &&
                      result.output_norm.numel() == c.hidden,
#else
    require_model(result.embedding_norm.sizes().vec() == std::vector<int64_t>{c.hidden} &&
                      result.hidden_norm.sizes().vec() == std::vector<int64_t>{c.hidden} &&
                      result.output_norm.numel() == c.hidden,
#endif
                  "GLM5-Next MTP normalization width disagrees with backbone");
    return true;
}

template <class Project, class Cache, class Absorb, class Dense, class Pool, class IndexQuery,
          class Score, class Select, class Sparse, class Unembed, class Output, class Rollback>
auto sparse_mla(int64_t length, int64_t budget, bool cache, Project project, Cache append,
                Absorb absorb, Dense dense, Pool pool, IndexQuery index_query, Score score,
                Select select, Sparse sparse, Unembed unembed, Output output, Rollback rollback) {
    auto projected = project();
    try {
        if (cache)
            append(projected);
        auto query = absorb(projected);
        auto attended = [&] {
            if (length <= budget)
                return dense(query, projected);
            auto pooled = pool(projected);
            auto iq = index_query(projected);
            auto scores = score(std::move(iq), std::move(pooled));
            auto selected = select(std::move(scores));
            return sparse(query, projected, std::move(selected));
        }();
        return output(unembed(std::move(attended)));
    } catch (...) {
        if (cache)
            rollback();
        throw;
    }
}

inline void load_config(std::string_view payload, Config& config, models::CausalLmMetadata& metadata) {
    config = Config::from_json(payload);
    metadata.vocab_size = config.vocab;
    metadata.hidden_size = config.hidden;
    metadata.num_hidden_layers = config.layers;
    metadata.num_attention_heads = config.heads;
    metadata.num_key_value_heads = 1;
    metadata.head_dim = config.nope;
    metadata.max_position_embeddings = config.maximum;
    metadata.num_experts = config.experts;
    metadata.hc_mult = config.streams;
    metadata.rope_base = 1.0;
    metadata.rms_norm_eps = config.eps;
    metadata.hc_eps = config.hc_eps;
    metadata.tie_word_embeddings = config.tied_embeddings;
    metadata.flash_next = true;
    metadata.model_type = "glm5_next";
    metadata.layer_types = config.layer_types;
}

} // namespace tyloi::models::glm5_next
