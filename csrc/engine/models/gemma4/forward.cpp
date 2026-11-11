#include "engine/models/gemma4/forward.h"
#include "engine/models/common/check.h"
#include "engine/ops.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda::gemma4 {

static void trace_stage(const Model::Options& mode, int layer, const char* name,
                        const tyloi::Tensor& value) {
    if (mode.trace != nullptr && layer == mode.trace_layer) {
        auto converted = tyloi::ops::cast(value, tyloi::ScalarType::float32);
        auto contiguous = tyloi::ops::contiguous(converted);
        mode.trace->emplace_back(name, tyloi::ops::clone(contiguous));
    }
}

static ::tyloi::Tensor gemma_rms_norm_f16(
    const tyloi::Tensor& x, const tyloi::Tensor& weight, double eps, double weight_offset) {
    tyloi::models::require_model(x.scalar_type() == tyloi::ScalarType::float16,
                                 "gemma_rms_norm_f16: activation must remain f16");
    return tyloi::ops::rms_norm(x, weight, eps, weight_offset, tyloi::ScalarType::float16);
}

static ::tyloi::Tensor forward_ffn(
    const Layer& weights, const Model::Options& mode, const tyloi::Tensor& residual,
    const tyloi::Tensor& oo, int64_t B, int64_t T, int64_t H) {
    namespace ops = tyloi::ops;
    const auto& attention = weights.attention;
    const auto layer = attention.layer;
    const auto eps = attention.rms_norm_eps;
    const auto offset = attention.norm_weight_offset;
    const auto profile_device = residual.device();
    auto measure = [&](const char* name, auto&& operation) {
        return ops::measure(name, profile_device, std::forward<decltype(operation)>(operation));
    };
    const std::array<std::int64_t, 2> flat_shape{B * T, H};
    const std::array<std::int64_t, 3> layer_shape{B, T, H};
    trace_stage(mode, layer, "attention_output", oo);
    ::tyloi::Tensor prepared_residual, dense_input, router_input, moe_input;
    bool fused_norms = false;
    if (weights.gemma4_moe && mode.trace == nullptr && weights.layer_scale && weights.layer_scale.defined()) {
        auto prepared = ops::residual_norms(residual, oo, weights.attn_post_norm, weights.ffn_norm,
                                           weights.gemma_router_norm_scale, weights.ffn_pre_norm_2, eps);
        if (prepared) {
            fused_norms = true;
            prepared_residual = ops::reshape(prepared->residual, layer_shape);
            dense_input = prepared->dense;
            router_input = prepared->router;
            moe_input = prepared->experts;
        }
    }
    if (!fused_norms) {
        auto attn_post = measure("gemma.attn_post_norm", [&] {
            auto flat = ops::reshape(oo, flat_shape);
            return gemma_rms_norm_f16(flat, weights.attn_post_norm, eps, offset);
        });
        prepared_residual = measure("gemma.attn_residual", [&] {
            auto flat = ops::reshape(residual, flat_shape);
            auto added = ops::add_residual(flat, attn_post);
            return ops::reshape(added, layer_shape);
        });
        trace_stage(mode, layer, "attention_residual", prepared_residual);
        dense_input = measure("gemma.ffn_pre_norm", [&] {
            auto flat = ops::reshape(prepared_residual, flat_shape);
            return gemma_rms_norm_f16(flat, weights.ffn_norm, eps, offset);
        });
        if (weights.gemma4_moe) {
            router_input = measure("gemma.router_norm", [&] {
                auto flat = ops::reshape(prepared_residual, flat_shape);
                auto converted = ops::cast(flat, tyloi::ScalarType::float32);
                return ops::rms_norm(converted, weights.gemma_router_norm_scale, eps, offset);
            });
            moe_input = measure("gemma.ffn_pre_norm_2", [&] {
                auto flat = ops::reshape(prepared_residual, flat_shape);
                return gemma_rms_norm_f16(flat, weights.ffn_pre_norm_2, eps, offset);
            });
        }
    }
    auto normalize = [&](const tyloi::Tensor& value, const tyloi::Tensor& weight, const char* name) {
        return measure(name, [&] {
            return ops::rms_norm(value, weight, eps, offset, tyloi::ScalarType::float16);
        });
    };
    auto result = measure("gemma.ffn_dense", [&] {
        auto dense = tyloi::models::mlp(weights.ffn, dense_input);
        return ops::reshape(dense, flat_shape);
    });
    if (weights.gemma4_moe) {
        if (!fused_norms) {
            result = normalize(result, weights.ffn_post_norm_1, "gemma.ffn_post_norm_1");
            trace_stage(mode, layer, "dense_output", result);
        }
        auto router_logits = measure("gemma.router", [&] {
            auto transposed = ops::transpose(weights.gemma_router, 0, 1);
            return ops::matmul(router_input, transposed);
        });
        auto selected = measure("gemma.topk", [&] {
            return ops::topk(router_logits, weights.gemma_top_k,
                             {.activation = ops::RouterActivation::softmax, .delayed_softmax = true});
        });
        trace_stage(mode, layer, "route_ids", selected.ids);
        trace_stage(mode, layer, "route_weights_before_scale", selected.weights);
        {
            ops::ProfileRange range("gemma.route_scale", profile_device);
            ops::expert_scale(selected.weights, selected.ids, weights.gemma_expert_scale);
            range.finish();
        }
        trace_stage(mode, layer, "route_weights", selected.weights);
        const bool tracing = mode.trace != nullptr && layer == mode.trace_layer;
        auto projected = ops::routed_geglu(moe_input, selected.ids,
                                           weights.gemma_moe_gate_up, weights.gemma_moe_down, tracing);
        if (tracing) {
            mode.trace->emplace_back("moe_gate_up", projected.gate_up);
            mode.trace->emplace_back("moe_hidden", projected.hidden);
        }
        trace_stage(mode, layer, "moe_down", projected.pairs);
        auto moe_output = measure("gemma.moe_reduce", [&] {
            return ops::weighted_reduce(projected.pairs, selected.weights);
        });
        trace_stage(mode, layer, "moe_reduce", moe_output);
        if (fused_norms) {
            result = measure("gemma.ffn_merge", [&] {
                auto flat = ops::reshape(prepared_residual, flat_shape);
                auto merged = ops::merge_norms(result, moe_output, flat, weights.ffn_post_norm_1,
                                                weights.ffn_post_norm_2, weights.ffn_post_norm,
                                                weights.layer_scale, eps);
                return ops::reshape(merged, layer_shape);
            });
        } else {
            auto normalized = normalize(moe_output, weights.ffn_post_norm_2, "gemma.ffn_post_norm_2");
            result = measure("gemma.ffn_combine", [&] { return ops::add(result, normalized); });
        }
    }
    if (!fused_norms) {
        auto normalized = normalize(result, weights.ffn_post_norm, "gemma.ffn_post_norm");
        result = measure("gemma.ffn_residual", [&] {
            auto flat = ops::reshape(prepared_residual, flat_shape);
            auto added = ops::add_residual(flat, normalized);
            return ops::reshape(added, layer_shape);
        });
        if (weights.layer_scale && weights.layer_scale.defined())
            result = measure("gemma.layer_scale", [&] { return ops::multiply(result, weights.layer_scale); });
    }
    trace_stage(mode, layer, "layer_output", result);
    return result;
}

namespace {

::tyloi::Tensor normalize_ids(int primary, const tyloi::Tensor& ids) {
    tyloi::models::require_model(ids.dim() == 1 || ids.dim() == 2,
                               "token IDs must have shape [tokens] or [batch,tokens]");
    auto shaped = ids.dim() == 1 ? tyloi::ops::reshape(ids, {1, ids.size(0)})
                                 : tyloi::ops::copy(ids, ids.device());
    tyloi::models::require_model(shaped.size(0) > 0 && shaped.size(1) > 0,
                               "token IDs must not be empty");
    auto converted = tyloi::ops::cast(shaped, tyloi::ScalarType::int64);
    auto copied = tyloi::ops::copy(converted, tyloi::Device{tyloi::DeviceType::cuda, primary});
    return tyloi::ops::contiguous(copied);
}

} // namespace

::tyloi::Tensor Model::embedding(const tyloi::Tensor& ids, int primary) const {
    auto normalized = normalize_ids(primary, ids);
    return tyloi::ops::measure("model.embed", normalized.device(), [&] {
        return tyloi::ops::embedding(normalized, embed);
    });
}

::tyloi::Tensor Model::normalize(const tyloi::Tensor& hidden, std::int64_t batch, std::int64_t tokens) const {
    const auto& model = *this;
    const std::array<std::int64_t, 3> shape{batch, tokens, model.config.hidden_size};
    auto shaped = tyloi::ops::reshape(hidden, shape);
    return tyloi::ops::measure("model.output_norm", shaped.device(), [&] {
        return tyloi::ops::rms_norm(shaped, model.output_norm,
                                       model.config.rms_norm_eps, 0.0);
    });
}

::tyloi::Tensor Model::softcap(const tyloi::Tensor& logits) const {
    return tyloi::ops::softcap(logits, config.final_logit_softcapping);
}

::tyloi::Tensor Model::logits(const tyloi::Tensor& hidden, bool capped) const {
    const auto& model = *this;
    auto result = tyloi::ops::measure("model.lm_head", hidden.device(), [&] {
        return tyloi::ops::project(hidden, model.lm_head);
    });
    return capped ? tyloi::ops::softcap(result, model.config.final_logit_softcapping)
                  : std::move(result);
}

::tyloi::Tensor forward_layer(const Layer& layer, tyloi::state::KvView& cache,
                              const tyloi::Tensor& input, const AttentionStep& step,
                              const Model::Options& mode) {
    const auto B = input.size(0), T = input.size(1), H = input.size(2);
    const auto& attention = layer.attention;
    const auto capacity = attention.sliding ? attention.attention_window
        : (mode.kv_capacity > 0 ? std::max<std::int64_t>(step.cache_position + T, mode.kv_capacity)
                                : attention.max_position_embeddings);
    if (!cache.defined()) {
        cache = tyloi::state::allocate_kv(B, attention.kv_heads, capacity, attention.attention_head_dim,
                        attention.sliding, input.device(),
                        attention.official_bf16 ? tyloi::ScalarType::bfloat16 : tyloi::ScalarType::float16,
                        attention.kv_quantization);
    }
    cache.quantization = attention.kv_quantization;
    auto result = attention.forward(input, step,
        cache, attention.attention_rope,
        attention.layer == mode.trace_layer ? mode.trace : nullptr);
    return forward_ffn(layer, mode,
                       result[0], result[1], B, T, H);
}

tyloi::StepSequence<tyloi::models::Prepared> Model::prepare(
    const std::vector<std::int64_t>&, const MfqMultimodalInput&, tyloi::models::State&) const {
    throw std::invalid_argument("model has no multimodal component");
}

::tyloi::Tensor Model::forward(tyloi::models::Input input, tyloi::models::State& state_value,
                                              const Options& mode) const {
    tyloi::models::require_model(!input.suffix, "Gemma4 does not support speculative suffix forwarding");
    auto& state = dynamic_cast<State&>(state_value);
    tyloi::models::require_model(state.belongs_to(*this),
                               "Gemma4 state belongs to a different model");
    const auto& model = *this;
    tyloi::models::require_model(input.ids.defined(),
                               "token IDs must have shape [tokens] or [batch,tokens]");
    auto ids = normalize_ids(mode.primary, input.ids);
    ::tyloi::Tensor embeddings = input.embeddings
        ? input.embeddings : embedding(ids, mode.primary);
    const auto B = ids.size(0), T = ids.size(1);
    tyloi::models::require_model(embeddings.dim() == 3 && embeddings.size(0) == B &&
                                   embeddings.size(1) == T &&
                                   embeddings.size(2) == model.config.hidden_size,
                               "inputs_embeds shape must match [batch,tokens,hidden_size]");
    tyloi::models::require_model(input.confirmed >= 0 && input.confirmed == 0,
                               "unsupported speculative backbone geometry");
    if (state.cache_pos == 0) state.reset(B);

    const tyloi::Device primary{tyloi::DeviceType::cuda, mode.primary};
    auto positions_on_primary = [&](const tyloi::Tensor& value) {
        auto converted = tyloi::ops::cast(value, tyloi::ScalarType::int64);
        auto copied = tyloi::ops::copy(converted, primary);
        return tyloi::ops::contiguous(copied);
    };
    auto physical = input.cache_positions ? positions_on_primary(input.cache_positions)
        : tyloi::ops::arange(state.cache_pos, state.cache_pos + T, primary);
    auto semantic = input.positions ? positions_on_primary(input.positions)
        : (state.decode_position_delta == 0 ? physical
            : tyloi::ops::add(physical, state.decode_position_delta));
    tyloi::models::require_model((physical.dim() == 1 && physical.numel() == T) ||
                                   (physical.dim() == 2 && physical.size(0) == B && physical.size(1) == T),
                               "cache_positions must have shape [tokens] or [batch,tokens]");
    tyloi::models::require_model((semantic.dim() == 1 && semantic.size(0) == T) ||
                                   (semantic.dim() == 2 && semantic.size(0) == B && semantic.size(1) == T),
                               "position_ids must have shape [tokens] or [batch,tokens]");
    if (!input.lengths && state.cache_pos > 0 && T == 1) {
        const std::array<std::int64_t, 1> shape{B};
        input.lengths = tyloi::ops::full(shape, state.cache_pos + 1, primary);
    }
    AttentionStep step{semantic, input.lengths, input.cache_positions ? physical : ::tyloi::Tensor{},
                       state.cache_pos, input.kv_length, input.attention_parts};
    auto copied = tyloi::ops::copy(embeddings, primary);
    auto hidden = tyloi::ops::contiguous(copied);
    hidden = tyloi::ops::measure("model.embed_scale", hidden.device(), [&] {
        return tyloi::ops::scale(hidden, model.embed_scale);
    });
    const auto trace = [&](const tyloi::Tensor& value) {
        if (input.trace) {
            auto local = tyloi::ops::copy(value, primary);
            auto converted = tyloi::ops::cast(local, tyloi::ScalarType::float32);
            input.trace->push_back(tyloi::ops::clone(converted));
        }
    };
    trace(hidden);
    for (std::size_t index = 0; index < model.layers.size(); ++index) {
        const auto& layer = model.layers[index];
        auto local = step;
        const tyloi::Device device{tyloi::DeviceType::cuda, layer.device};
        local.positions = tyloi::ops::copy(step.positions, device);
        if (step.sequence_lengths)
            local.sequence_lengths = tyloi::ops::copy(step.sequence_lengths, device);
        if (step.cache_positions)
            local.cache_positions = tyloi::ops::copy(step.cache_positions, device);
        auto local_hidden = tyloi::ops::copy(hidden, device);
        hidden = forward_layer(layer, state.layers[index], local_hidden, local, mode);
        trace(hidden);
    }
    if (!input.positions || input.advance) state.cache_pos += T;
    hidden = tyloi::ops::copy(hidden, primary);
    auto finalized = normalize(hidden, B, T);
    if (input.raw_hidden) *input.raw_hidden = std::move(hidden);
    return finalized;
}

} // namespace tyloi::cuda::gemma4
