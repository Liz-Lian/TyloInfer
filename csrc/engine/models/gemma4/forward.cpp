#include "engine/models/gemma4/forward.h"
#include "engine/models/gemma4/ffn.h"
#include "engine/models/common/check.h"
#include "engine/ops.h"
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/kernels/mfq_cuda_moe_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#endif

#include <algorithm>
#include <array>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

namespace mfq::cuda::gemma4 {

struct FfnStep {
    Ffn::Mode ffn;
    Trace* trace;
    int trace_layer;
    bool fused_norms;
};

static void trace_stage(const FfnStep& mode, int layer, const char* name,
                        const Tensor& value) {
    if (mode.trace != nullptr && layer == mode.trace_layer) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        mode.trace->emplace_back(name, Tensor(mfq::ops::clone(mfq::ops::contiguous(
            mfq::ops::cast(value, mfq::ScalarType::float32)))));
#else
        mode.trace->emplace_back(name, value.to(mfq_tensor_backend::kFloat32).contiguous().clone());
#endif
    }
}

static mfq_tensor_backend::Tensor gemma_rms_norm_f16(mfq_tensor_backend::Tensor x,
                                                     mfq_tensor_backend::Tensor weight, double eps,
                                                     double weight_offset) {
    TYLOINFER_RUNTIME_CHECK(x.scalar_type() == mfq_tensor_backend::kFloat16,
                      "gemma_rms_norm_f16: activation must remain f16");
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    return Tensor(mfq::ops::rms_norm(x, weight, eps, weight_offset, mfq::ScalarType::float16));
#else
    return rms_norm_f16_cuda(x.contiguous(), weight, eps, weight_offset);
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
static Tensor forward_ffn(const Layer& weights, const FfnStep& mode,
#else
static Tensor forward_ffn(const Layer& weights, Executor& executor, const FfnStep& mode,
#endif
                          Tensor residual, Tensor oo, int64_t B, int64_t T, int64_t H) {
    const auto& gemma4_moe = weights.gemma4_moe;
    const auto& attn_post_norm = weights.attn_post_norm;
    const auto& ffn_norm = weights.ffn_norm;
    const auto& ffn_post_norm = weights.ffn_post_norm;
    const auto& ffn_post_norm_1 = weights.ffn_post_norm_1;
    const auto& ffn_pre_norm_2 = weights.ffn_pre_norm_2;
    const auto& ffn_post_norm_2 = weights.ffn_post_norm_2;
    const auto& layer_scale = weights.layer_scale;
    const auto& gemma_moe_gate_up = weights.gemma_moe_gate_up;
    const auto& gemma_moe_down = weights.gemma_moe_down;
    const auto& gemma_router = weights.gemma_router;
    const auto& gemma_router_norm_scale = weights.gemma_router_norm_scale;
    const auto& gemma_expert_scale = weights.gemma_expert_scale;
    const auto& gemma_top_k = weights.gemma_top_k;
    const auto& ffn = weights.ffn;
    const auto layer = weights.attention.layer;
    const auto rms_norm_eps = weights.attention.rms_norm_eps;
    const auto norm_weight_offset = weights.attention.norm_weight_offset;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    using mfq::ops::measure;
#else
    auto measure = [&](const char* name, auto&& operation) {
        return executor.profiler().measure(name, std::forward<decltype(operation)>(operation));
    };
#endif
    trace_stage(mode, layer, "attention_output", oo);
    const bool fused_norms = gemma4_moe && mode.fused_norms &&
                             mode.trace == nullptr && layer_scale.defined();
    using Tensor = mfq_tensor_backend::Tensor;
    using Inputs = mfq::models::gemma4::FfnInputs<Tensor>;
    struct Selection { Tensor ids, weights; };
    auto inputs = mfq::models::gemma4::prepare_ffn(
        residual, oo, gemma4_moe,
        [&](Tensor residual, Tensor oo) -> std::optional<Inputs> {
            if (!fused_norms)
                return {};
            auto prepared = measure("gemma.attn_residual_pre_norms", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                auto values = mfq::ops::residual_norms(
                    residual.reshape({B * T, H}), oo.reshape({B * T, H}), attn_post_norm, ffn_norm,
                    gemma_router_norm_scale, ffn_pre_norm_2, rms_norm_eps);
                return std::array<Tensor, 4>{Tensor(values.residual), Tensor(values.dense),
                                             Tensor(values.router), Tensor(values.experts)};
#else
                return gemma4_attn_residual_pre_norms_f16_cuda(
                    residual.reshape({B * T, H}), oo.reshape({B * T, H}), attn_post_norm, ffn_norm,
                    gemma_router_norm_scale, ffn_pre_norm_2, rms_norm_eps);
#endif
            });
            return Inputs{prepared[0].reshape({B, T, H}), prepared[1], prepared[2], prepared[3]};
        },
        [&](Tensor oo) {
            return measure("gemma.attn_post_norm", [&]() {
                return gemma_rms_norm_f16(oo.reshape({B * T, H}), attn_post_norm, rms_norm_eps,
                                          norm_weight_offset);
            });
        },
        [&](Tensor residual, Tensor attn_post) {
            auto x = measure("gemma.attn_residual", [&] {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                return Tensor(mfq::ops::add_residual(residual.reshape({B * T, H}), attn_post)).reshape({B, T, H});
#else
                return acc_cuda(residual.reshape({B * T, H}), attn_post).reshape({B, T, H});
#endif
            });
            trace_stage(mode, layer, "attention_residual", x);
            return x;
        },
        [&](Tensor x) {
            return measure("gemma.ffn_pre_norm", [&] {
                return gemma_rms_norm_f16(x.reshape({B * T, H}), ffn_norm, rms_norm_eps,
                                          norm_weight_offset);
            });
        },
        [&](Tensor x) {
            return measure("gemma.router_norm", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                return Tensor(mfq::ops::rms_norm(
                    mfq::ops::cast(x.reshape({B * T, H}), mfq::ScalarType::float32),
                    gemma_router_norm_scale, rms_norm_eps, norm_weight_offset));
#else
                return qwen_rms_norm(x.reshape({B * T, H}).to(mfq_tensor_backend::kFloat32),
                                     gemma_router_norm_scale, rms_norm_eps, norm_weight_offset);
#endif
            });
        },
        [&](Tensor x) {
            return measure("gemma.ffn_pre_norm_2", [&]() {
                return gemma_rms_norm_f16(x.reshape({B * T, H}), ffn_pre_norm_2, rms_norm_eps,
                                          norm_weight_offset);
            });
        });
    residual = inputs.residual;
    auto dense_input = inputs.dense, router_input = inputs.router, moe_input = inputs.experts;
    auto result = mfq::models::gemma4::feed_forward(
        gemma4_moe, layer_scale.defined(), fused_norms, residual,
        [&] {
            return measure("gemma.ffn_dense", [&] {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                return ffn.forward(dense_input, mode.ffn).reshape({B * T, H});
#else
                return ffn.forward(dense_input, executor, mode.ffn).reshape({B * T, H});
#endif
            });
        },
        [&] {
#if TYLOINFER_ENABLE_MFQ_COMPAT
            return mfq::models::gemma4::experts(
                [&] {
                    return measure("gemma.router", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                        return Tensor(mfq::ops::matmul(router_input, mfq::ops::transpose(gemma_router, 0, 1)));
#else
                        return mfq_tensor_backend::matmul(router_input, gemma_router.transpose(0, 1));
#endif
                    });
                },
                [&](Tensor router_logits) {
                    return measure("gemma.topk", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                        auto selected = mfq::ops::topk(router_logits, gemma_top_k,
                            {.activation = mfq::ops::RouterActivation::softmax, .delayed_softmax = true});
                        return Selection{Tensor(selected.ids), Tensor(selected.weights)};
#else
                        auto selected = moe_topk_cuda(router_logits.contiguous(), gemma_top_k, false, false,
                                                      false, true, mfq_nullopt, 1e-20, 1.0);
                        return Selection{selected.at(0), selected.at(1)};
#endif
                    });
                },
                [&](auto &selected) {
                    trace_stage(mode, layer, "route_ids", selected.ids);
                    trace_stage(mode, layer, "route_weights_before_scale",
                                      selected.weights);
                    measure("gemma.route_scale", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                        return Tensor(mfq::ops::expert_scale(selected.weights, selected.ids, gemma_expert_scale));
#else
                        return moe_apply_expert_scale_cuda(selected.weights, selected.ids,
                                                           gemma_expert_scale);
#endif
                    });
                    trace_stage(mode, layer, "route_weights", selected.weights);
                },
                [&](const auto &selected) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                    const bool tracing = mode.trace != nullptr && layer == mode.trace_layer;
                    auto result = mfq::ops::routed_geglu(moe_input, selected.ids,
                        gemma_moe_gate_up, gemma_moe_down,
                        {mode.ffn.serial, mode.ffn.projection_major}, tracing);
                    if (tracing) {
                        mode.trace->emplace_back("moe_gate_up", Tensor(result.gate_up));
                        mode.trace->emplace_back("moe_hidden", Tensor(result.hidden));
                    }
                    Tensor down_pair(result.pairs);
                    trace_stage(mode, layer, "moe_down", down_pair);
                    return down_pair;
#else
                    auto route = measure("gemma.route_map", [&]() {
                        return build_moe_route_plan(selected.ids, gemma_moe_gate_up.n_experts);
                    });
                    const bool projection_bundle_prefetched = prefetch_cached_moe_projection_bundle(
                        gemma_moe_gate_up, gemma_moe_down, route);
                    const bool tracing = mode.trace != nullptr &&
                                         layer == mode.trace_layer;
                    Tensor packed;
                    auto prefetch_down = [&] {
                        if (!projection_bundle_prefetched)
                            mfq::cuda::prefetch_experts(gemma_moe_down, route);
                    };
                    auto down_pair = mfq::ops::gated_mlp(
                        moe_input, true, 0.0,
                        [](const auto &...) { return std::optional<Tensor>{}; },
                        [&](const Tensor &input, auto, double) -> std::optional<Tensor> {
                            if (tracing || !gemma_moe_gate_up.supports_projection_glu_epilogue())
                                return {};
                            auto hidden = measure("gemma.moe_gate_up_geglu", [&] {
                                return executor.projector(mode.ffn).glu(gemma_moe_gate_up, input, route, true);
                            });
                            prefetch_down();
                            return hidden;
                        },
                        [&](Tensor input) {
                            packed = measure("gemma.moe_gate_up", [&] {
                                return executor.projector(mode.ffn).run(gemma_moe_gate_up, input, route);
                            });
                            prefetch_down();
                            trace_stage(mode, layer, "moe_gate_up", packed);
                            const auto width = packed.size(-1) / 2;
                            return std::array<Tensor, 2>{packed.narrow(-1, 0, width),
                                                         packed.narrow(-1, width, width)};
                        },
                        [&](const Tensor &, const Tensor &, auto, double) {
                            auto hidden = measure(
                                "gemma.moe_geglu", [&] { return moe_geglu_split_cuda(packed); });
                            if (tracing)
                                trace_stage(mode, layer, "moe_hidden", hidden);
                            return hidden;
                        },
                        [&](Tensor hidden) {
                            return measure("gemma.moe_down", [&] {
                                return executor.projector(mode.ffn).run(gemma_moe_down, hidden, route);
                            });
                        },
                        [&](const Tensor &, const Tensor &, auto, double) -> std::optional<Tensor> {
                            if (tracing || packed.size(0) > 4)
                                return {};
                            return measure("gemma.moe_geglu_down", [&] {
                                return executor.projector(mode.ffn).gated(gemma_moe_down, packed, route, true);
                            });
                        });
                    trace_stage(mode, layer, "moe_down", down_pair);
                    return down_pair;
#endif
                },
                [&](Tensor down_pair, const auto &selected) {
                    auto moe_output = measure("gemma.moe_reduce", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                        return Tensor(mfq::ops::weighted_reduce(down_pair, selected.weights));
#else
                        return moe_weighted_reduce_cuda(down_pair, selected.weights);
#endif
                    });
                    trace_stage(mode, layer, "moe_reduce", moe_output);
                    return moe_output;
                });
#else
            throw std::runtime_error("Legacy Gemma MoE weights require a compatibility build");
            return Tensor{};
#endif
        },
        [&](Tensor value, mfq::models::gemma4::FfnNorm role) {
            using Norm = mfq::models::gemma4::FfnNorm;
            const auto &weight = role == Norm::dense     ? ffn_post_norm_1
                                 : role == Norm::experts ? ffn_post_norm_2
                                                         : ffn_post_norm;
            const char *name = role == Norm::dense     ? "gemma.ffn_post_norm_1"
                               : role == Norm::experts ? "gemma.ffn_post_norm_2"
                                                       : "gemma.ffn_post_norm";
            auto normalized = measure(name, [&] {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                return Tensor(mfq::ops::rms_norm(value, weight, rms_norm_eps,
                                                  norm_weight_offset, mfq::ScalarType::float16));
#else
                return value.scalar_type() == mfq_tensor_backend::kFloat16
                           ? gemma_rms_norm_f16(value, weight, rms_norm_eps, norm_weight_offset)
                           : qwen_rms_norm(value.to(mfq_tensor_backend::kFloat32), weight,
                                           rms_norm_eps, norm_weight_offset)
                                 .to(mfq_tensor_backend::kFloat16)
                                 .contiguous();
#endif
            });
            if (role == Norm::dense)
                trace_stage(mode, layer, "dense_output", normalized);
            return normalized;
        },
        [&](Tensor dense, Tensor routed) {
            return measure("gemma.ffn_combine", [&] {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                return Tensor(mfq::ops::add(dense, routed));
#else
                return dense + routed;
#endif
            });
        },
        [&](Tensor skip, Tensor value) {
            return measure("gemma.ffn_residual", [&] {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                return Tensor(mfq::ops::add_residual(skip.reshape({B * T, H}), value)).reshape({B, T, H});
#else
                return acc_cuda(skip.reshape({B * T, H}), value).reshape({B, T, H});
#endif
            });
        },
        [&](Tensor value) {
            return measure("gemma.layer_scale", [&] {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                return Tensor(mfq::ops::multiply(value, layer_scale));
#else
                return value * layer_scale;
#endif
            });
        },
        [&](Tensor dense, Tensor routed, Tensor skip) {
            return measure("gemma.ffn_merge", [&] {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                return Tensor(mfq::ops::merge_norms(dense, routed, skip.reshape({B * T, H}),
                    ffn_post_norm_1, ffn_post_norm_2, ffn_post_norm, layer_scale, rms_norm_eps))
                    .reshape({B, T, H});
#else
                return gemma4_ffn_merge_f16_cuda(dense, routed, skip.reshape({B * T, H}),
                                                 ffn_post_norm_1, ffn_post_norm_2, ffn_post_norm,
                                                 layer_scale, rms_norm_eps)
                    .reshape({B, T, H});
#endif
            });
        });
    trace_stage(mode, layer, "layer_output", result);
    return result;
}

namespace {

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
Tensor normalize_ids(int primary, Tensor ids) {
#else
Tensor normalize_ids(Collectives& collectives, int primary, Tensor ids) {
#endif
    mfq::models::require_model(ids.dim() == 1 || ids.dim() == 2,
                               "token IDs must have shape [tokens] or [batch,tokens]");
    if (ids.dim() == 1) ids = ids.unsqueeze(0);
    mfq::models::require_model(ids.size(0) > 0 && ids.size(1) > 0,
                               "token IDs must not be empty");
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    return Tensor(mfq::ops::contiguous(mfq::ops::copy(
        mfq::ops::cast(std::move(ids), mfq::ScalarType::int64),
        mfq::Device{mfq::DeviceType::cuda, primary})));
#else
    return collectives.copy(ids.to(mfq_tensor_backend::kInt64), primary).contiguous();
#endif
}

} // namespace

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
mfq::Tensor Model::embedding(mfq::Tensor ids, int primary) const {
    auto normalized = normalize_ids(primary, Tensor(std::move(ids)));
    return mfq::ops::measure("model.embed", [&] {
        return mfq::ops::embedding(std::move(normalized), *embed);
    });
#else
Tensor embed(const Model& model, Executor& executor, Collectives& collectives,
             int primary, Tensor ids, const Executor::Mode& mode) {
    ids = normalize_ids(collectives, primary, std::move(ids));
    const Task task{task::Embed{0, 0, as_quant_linear(*model.embed)}, -1, {}, "model.embed"};
    Inputs inputs;
    inputs.values = {std::move(ids)};
    return executor.run(std::span<const Task>{&task, 1}, std::move(inputs), mode);
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
mfq::Tensor Model::normalize(mfq::Tensor hidden, std::int64_t batch, std::int64_t tokens) const {
    const auto& model = *this;
    const std::array<std::int64_t, 3> shape{batch, tokens, model.config.hidden_size};
    hidden = mfq::ops::reshape(std::move(hidden), shape);
    return mfq::ops::measure("model.output_norm", [&] {
        return mfq::ops::rms_norm(std::move(hidden), model.output_norm,
                                       model.config.rms_norm_eps, 0.0);
    });
#else
Tensor normalize(const Model& model, Executor& executor, Tensor hidden,
                 std::int64_t batch, std::int64_t tokens) {
    hidden = hidden.reshape({batch, tokens, model.config.hidden_size});
    const Task task{task::Norm{0, 0, model.output_norm, model.config.rms_norm_eps, 0.0},
                    -1, {}, "model.output_norm"};
    Inputs inputs;
    inputs.values = {std::move(hidden)};
    return executor.run(std::span<const Task>{&task, 1}, std::move(inputs), {});
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
mfq::Tensor Model::softcap(mfq::Tensor logits) const {
    return mfq::ops::softcap(std::move(logits), config.final_logit_softcapping);
#else
Tensor softcap(const Model& model, Executor& executor, Tensor logits) {
    const Task task{task::Softcap{0, 0, model.config.final_logit_softcapping}};
    Inputs inputs;
    inputs.values = {std::move(logits)};
    return executor.run(std::span<const Task>{&task, 1}, std::move(inputs), {});
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
mfq::Tensor Model::logits(mfq::Tensor hidden, bool capped, const Options& mode) const {
    const auto& model = *this;
    auto result = mfq::ops::measure("model.lm_head", [&] {
        return mfq::ops::project(std::move(hidden), *model.lm_head,
                                 {mode.ffn.serial, mode.ffn.projection_major});
    });
    return capped ? mfq::ops::softcap(std::move(result), model.config.final_logit_softcapping)
                  : std::move(result);
#else
Tensor logits(const Model& model, Executor& executor, Tensor hidden, bool capped,
              const Executor::Mode& mode) {
    std::array<Task, 2> tasks{
        Task{task::Project{0, 0, as_quant_linear(*model.lm_head)}, -1, {}, "model.lm_head"},
        Task{task::Softcap{0, 0, model.config.final_logit_softcapping}}};
    Inputs inputs;
    inputs.values = {std::move(hidden)};
    return executor.run(std::span<const Task>{tasks.data(), capped ? tasks.size() : 1},
                        std::move(inputs), mode);
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
Tensor forward_layer(const Layer& layer, KVCache& cache, Tensor input, const AttentionStep& step,
                     const Model::Options& mode) {
#else
Tensor forward_layer(const Layer& layer, KVCache& cache, Executor& executor,
                     std::size_t scratch_index, Tensor input, const AttentionStep& step,
                     const Executor::Mode& mode) {
#endif
    MfqCudaGuard guard(layer.device);
    const auto B = input.size(0), T = input.size(1), H = input.size(2);
    const auto& attention = layer.attention;
    const auto capacity = attention.sliding ? attention.attention_window
        : (mode.kv_capacity > 0 ? std::max<std::int64_t>(step.cache_position + T, mode.kv_capacity)
                                : attention.max_position_embeddings);
    if (!cache.defined()) {
        cache = KVCache(B, attention.kv_heads, capacity, attention.attention_head_dim,
                        attention.sliding, input.device(),
                        attention.official_bf16 ? mfq_tensor_backend::kBFloat16
                                                : mfq_tensor_backend::kFloat16,
                        attention.kv_quantization);
    }
    cache.quantization = attention.kv_quantization;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    auto result = attention.forward(std::move(input), step, cache.view(), attention.attention_rope,
        {mode.ffn.serial, mode.ffn.projection_major},
        attention.layer == mode.trace_layer ? mode.trace : nullptr);
#else
    auto result = attention.forward(executor.projector(mode.ffn), executor.profiler(),
        executor.attention_options(), std::move(input), step, cache.view(),
        executor.attention_scratch(scratch_index), attention.attention_rope,
        attention.layer == mode.trace_layer ? mode.trace : nullptr);
#endif
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    return forward_ffn(layer, {mode.ffn, mode.trace, mode.trace_layer, mode.fused_norms},
                       std::move(result[0]), std::move(result[1]), B, T, H);
#else
    return forward_ffn(layer, executor, {mode.ffn, mode.trace, mode.trace_layer, mode.gemma4_fused_norms},
                       std::move(result[0]), std::move(result[1]), B, T, H);
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
mfq::StepSequence<mfq::models::Prepared> Model::prepare(
    const std::vector<std::int64_t>&, const MfqMultimodalInput&, mfq::models::State&) const {
    throw std::invalid_argument("model has no multimodal component");
}

mfq::Tensor Model::forward(mfq::models::Input input, mfq::models::State& state_value,
                           const Options& mode) const {
    auto& state = dynamic_cast<State&>(state_value);
    const auto& model = *this;
#else
Tensor forward(const Model& model, State& state, Executor& executor, Collectives& collectives,
               const Executor::Mode& mode, Input input) {
#endif
    namespace tb = mfq_tensor_backend;
    MfqCudaGuard guard(mode.primary);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    auto ids = normalize_ids(mode.primary, Tensor(std::move(input.ids)));
    auto embeddings = input.embeddings.has_value() ? std::move(*input.embeddings)
        : embedding(ids, mode.primary);
#else
    auto ids = normalize_ids(collectives, mode.primary, std::move(input.ids));
    auto embeddings = input.embeddings.has_value() ? std::move(*input.embeddings)
        : embed(model, executor, collectives, mode.primary, ids, mode);
#endif
    const auto B = ids.size(0), T = ids.size(1);
    mfq::models::require_model(embeddings.dim() == 3 && embeddings.size(0) == B &&
                                   embeddings.size(1) == T &&
                                   embeddings.size(2) == model.config.hidden_size,
                               "inputs_embeds shape must match [batch,tokens,hidden_size]");
    mfq::models::require_model(input.confirmed >= 0 && input.confirmed == 0,
                               "unsupported speculative backbone geometry");
    if (state.cache_pos == 0) state.reset(B);

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    const mfq::Device primary{mfq::DeviceType::cuda, mode.primary};
    auto physical = input.cache_positions.has_value()
        ? Tensor(mfq::ops::contiguous(mfq::ops::copy(
              mfq::ops::cast(*input.cache_positions, mfq::ScalarType::int64), primary)))
        : Tensor(mfq::ops::arange(state.cache_pos, state.cache_pos + T, primary));
    auto semantic = input.positions.has_value()
        ? Tensor(mfq::ops::contiguous(mfq::ops::copy(
              mfq::ops::cast(*input.positions, mfq::ScalarType::int64), primary)))
        : (state.decode_position_delta == 0 ? physical
            : Tensor(mfq::ops::add(physical, state.decode_position_delta)));
#else
    const auto options = tb::TensorOptions().device(tb::Device(tb::kCUDA, mode.primary))
                             .dtype(tb::kInt64);
    auto physical = input.cache_positions.has_value()
        ? collectives.copy(input.cache_positions->to(tb::kInt64), mode.primary).contiguous()
        : tb::arange(state.cache_pos, state.cache_pos + T, options);
    auto semantic = input.positions.has_value()
        ? collectives.copy(input.positions->to(tb::kInt64), mode.primary).contiguous()
        : (state.decode_position_delta == 0 ? physical : physical + state.decode_position_delta);
#endif
    mfq::models::require_model((physical.dim() == 1 && physical.numel() == T) ||
                                   (physical.dim() == 2 && physical.size(0) == B && physical.size(1) == T),
                               "cache_positions must have shape [tokens] or [batch,tokens]");
    mfq::models::require_model((semantic.dim() == 1 && semantic.size(0) == T) ||
                                   (semantic.dim() == 2 && semantic.size(0) == B && semantic.size(1) == T),
                               "position_ids must have shape [tokens] or [batch,tokens]");
    if (input.attention_mask.has_value()) {
        const auto& mask = *input.attention_mask;
        mfq::models::require_model(mask.dim() == 2 && mask.size(0) == B &&
                                       mask.size(1) >= state.cache_pos + T,
                                   "attention_mask must cover [batch,cached_plus_current_tokens]");
    }
    if (!input.lengths.has_value() && state.cache_pos > 0 && T == 1) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        const std::array<std::int64_t, 1> shape{B};
        input.lengths = Tensor(mfq::ops::full(shape, state.cache_pos + 1, primary));
#else
        input.lengths = tb::full({B}, state.cache_pos + 1, options);
#endif
    }
    AttentionStep step{semantic, input.lengths ? MfqOptional<Tensor>(Tensor(*input.lengths)) : mfq_nullopt,
                       input.cache_positions.has_value() ? MfqOptional<Tensor>(physical) : mfq_nullopt,
                       {}, state.cache_pos, input.kv_length, input.attention_parts};
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    auto hidden = Tensor(mfq::ops::contiguous(mfq::ops::copy(std::move(embeddings), primary)));
#else
    auto hidden = collectives.copy(std::move(embeddings), mode.primary).contiguous();
#endif
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    hidden = mfq::ops::measure("model.embed_scale", [&] {
        return Tensor(mfq::ops::scale(std::move(hidden), model.embed_scale));
#else
    hidden = executor.profiler().measure("model.embed_scale", [&] {
        return hidden * model.embed_scale;
#endif
    });
    const auto trace = [&](const Tensor& value) {
        if (input.trace) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            input.trace->push_back(Tensor(mfq::ops::clone(mfq::ops::cast(
                mfq::ops::copy(value, primary), mfq::ScalarType::float32))));
#else
            input.trace->push_back(collectives.copy(value, mode.primary).to(tb::kFloat32).clone());
#endif
        }
    };
    trace(hidden);
    for (std::size_t index = 0; index < model.layers.size(); ++index) {
        const auto& layer = model.layers[index];
        {
            MfqCudaGuard layer_guard(layer.device);
            auto local = step;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            const mfq::Device device{mfq::DeviceType::cuda, layer.device};
            local.positions = Tensor(mfq::ops::copy(step.positions, device));
            if (step.sequence_lengths.has_value())
                local.sequence_lengths = Tensor(mfq::ops::copy(*step.sequence_lengths, device));
            if (step.cache_positions.has_value())
                local.cache_positions = Tensor(mfq::ops::copy(*step.cache_positions, device));
            hidden = forward_layer(layer, state.layers[index],
                                   Tensor(mfq::ops::copy(std::move(hidden), device)), local, mode);
#else
            local.positions = collectives.copy(step.positions, layer.device);
            if (step.sequence_lengths.has_value())
                local.sequence_lengths = collectives.copy(*step.sequence_lengths, layer.device);
            if (step.cache_positions.has_value())
                local.cache_positions = collectives.copy(*step.cache_positions, layer.device);
            hidden = forward_layer(layer, state.layers[index], executor, index,
                                   collectives.copy(std::move(hidden), layer.device), local, mode);
#endif
        }
        trace(hidden);
    }
    if (!input.positions.has_value() || input.advance) state.cache_pos += T;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    hidden = Tensor(mfq::ops::copy(std::move(hidden), primary));
#else
    hidden = collectives.copy(std::move(hidden), mode.primary);
#endif
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    auto finalized = normalize(hidden, B, T);
#else
    auto finalized = normalize(model, executor, hidden, B, T);
#endif
    if (input.raw_hidden) *input.raw_hidden = std::move(hidden);
    return finalized;
}

} // namespace mfq::cuda::gemma4
