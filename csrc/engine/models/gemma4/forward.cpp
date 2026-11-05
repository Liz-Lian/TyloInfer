#include "engine/models/gemma4/forward.h"
#include "engine/models/common/check.h"
#include "engine/ops.h"
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/backends/cuda/native/tensor/backend.h"
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

namespace tyloi::cuda::gemma4 {

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
static void trace_stage(const Model::Options& mode, int layer, const char* name,
                        const tyloi::Tensor& value) {
#else
static void trace_stage(const Executor::Mode& mode, int layer, const char* name,
                        const DenseTensor& value) {
#endif
    if (mode.trace != nullptr && layer == mode.trace_layer) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        auto converted = tyloi::ops::cast(value, tyloi::ScalarType::float32);
        auto contiguous = tyloi::ops::contiguous(*converted);
        mode.trace->emplace_back(name, tyloi::ops::clone(*contiguous));
#else
        mode.trace->emplace_back(name, value.to(::tyloi::kFloat32).contiguous().clone());
#endif
    }
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
static std::shared_ptr<tyloi::Tensor> gemma_rms_norm_f16(
    const tyloi::Tensor& x, const tyloi::Tensor& weight, double eps, double weight_offset) {
    tyloi::models::require_model(x.scalar_type() == tyloi::ScalarType::float16,
                                 "gemma_rms_norm_f16: activation must remain f16");
    return tyloi::ops::rms_norm(x, weight, eps, weight_offset, tyloi::ScalarType::float16);
}

static std::shared_ptr<tyloi::Tensor> forward_ffn(
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
    std::shared_ptr<tyloi::Tensor> prepared_residual, dense_input, router_input, moe_input;
    bool fused_norms = false;
    if (weights.gemma4_moe && mode.trace == nullptr && weights.layer_scale.defined()) {
        auto prepared = ops::residual_norms(residual, oo, weights.attn_post_norm, weights.ffn_norm,
                                           weights.gemma_router_norm_scale, weights.ffn_pre_norm_2, eps);
        if (prepared) {
            fused_norms = true;
            prepared_residual = ops::reshape(*prepared->residual, layer_shape);
            dense_input = prepared->dense;
            router_input = prepared->router;
            moe_input = prepared->experts;
        }
    }
    if (!fused_norms) {
        auto attn_post = measure("gemma.attn_post_norm", [&] {
            auto flat = ops::reshape(oo, flat_shape);
            return gemma_rms_norm_f16(*flat, weights.attn_post_norm, eps, offset);
        });
        prepared_residual = measure("gemma.attn_residual", [&] {
            auto flat = ops::reshape(residual, flat_shape);
            auto added = ops::add_residual(*flat, *attn_post);
            return ops::reshape(*added, layer_shape);
        });
        trace_stage(mode, layer, "attention_residual", *prepared_residual);
        dense_input = measure("gemma.ffn_pre_norm", [&] {
            auto flat = ops::reshape(*prepared_residual, flat_shape);
            return gemma_rms_norm_f16(*flat, weights.ffn_norm, eps, offset);
        });
        if (weights.gemma4_moe) {
            router_input = measure("gemma.router_norm", [&] {
                auto flat = ops::reshape(*prepared_residual, flat_shape);
                auto converted = ops::cast(*flat, tyloi::ScalarType::float32);
                return ops::rms_norm(*converted, weights.gemma_router_norm_scale, eps, offset);
            });
            moe_input = measure("gemma.ffn_pre_norm_2", [&] {
                auto flat = ops::reshape(*prepared_residual, flat_shape);
                return gemma_rms_norm_f16(*flat, weights.ffn_pre_norm_2, eps, offset);
            });
        }
    }
    auto normalize = [&](const tyloi::Tensor& value, const tyloi::Tensor& weight, const char* name) {
        return measure(name, [&] {
            return ops::rms_norm(value, weight, eps, offset, tyloi::ScalarType::float16);
        });
    };
    auto result = measure("gemma.ffn_dense", [&] {
        auto dense = weights.ffn.forward(*dense_input);
        return ops::reshape(*dense, flat_shape);
    });
    if (weights.gemma4_moe) {
        if (!fused_norms) {
            result = normalize(*result, weights.ffn_post_norm_1, "gemma.ffn_post_norm_1");
            trace_stage(mode, layer, "dense_output", *result);
        }
#if TYLOINFER_ENABLE_MFQ_COMPAT
        auto router_logits = measure("gemma.router", [&] {
            auto transposed = ops::transpose(weights.gemma_router, 0, 1);
            return ops::matmul(*router_input, *transposed);
        });
        auto selected = measure("gemma.topk", [&] {
            return ops::topk(*router_logits, weights.gemma_top_k,
                             {.activation = ops::RouterActivation::softmax, .delayed_softmax = true});
        });
        trace_stage(mode, layer, "route_ids", *selected.ids);
        trace_stage(mode, layer, "route_weights_before_scale", *selected.weights);
        {
            ops::ProfileRange range("gemma.route_scale", profile_device);
            ops::expert_scale(*selected.weights, *selected.ids, weights.gemma_expert_scale);
            range.finish();
        }
        trace_stage(mode, layer, "route_weights", *selected.weights);
        const bool tracing = mode.trace != nullptr && layer == mode.trace_layer;
        auto projected = ops::routed_geglu(*moe_input, *selected.ids,
                                           *weights.gemma_moe_gate_up, *weights.gemma_moe_down, tracing);
        if (tracing) {
            mode.trace->emplace_back("moe_gate_up", projected.gate_up);
            mode.trace->emplace_back("moe_hidden", projected.hidden);
        }
        trace_stage(mode, layer, "moe_down", *projected.pairs);
        auto moe_output = measure("gemma.moe_reduce", [&] {
            return ops::weighted_reduce(*projected.pairs, *selected.weights);
        });
        trace_stage(mode, layer, "moe_reduce", *moe_output);
        if (fused_norms) {
            result = measure("gemma.ffn_merge", [&] {
                auto flat = ops::reshape(*prepared_residual, flat_shape);
                auto merged = ops::merge_norms(*result, *moe_output, *flat, weights.ffn_post_norm_1,
                                                weights.ffn_post_norm_2, weights.ffn_post_norm,
                                                weights.layer_scale, eps);
                return ops::reshape(*merged, layer_shape);
            });
        } else {
            auto normalized = normalize(*moe_output, weights.ffn_post_norm_2, "gemma.ffn_post_norm_2");
            result = measure("gemma.ffn_combine", [&] { return ops::add(*result, *normalized); });
        }
#else
        throw std::runtime_error("Legacy Gemma MoE weights require a compatibility build");
#endif
    }
    if (!fused_norms) {
        auto normalized = normalize(*result, weights.ffn_post_norm, "gemma.ffn_post_norm");
        result = measure("gemma.ffn_residual", [&] {
            auto flat = ops::reshape(*prepared_residual, flat_shape);
            auto added = ops::add_residual(*flat, *normalized);
            return ops::reshape(*added, layer_shape);
        });
        if (weights.layer_scale.defined())
            result = measure("gemma.layer_scale", [&] { return ops::multiply(*result, weights.layer_scale); });
    }
    trace_stage(mode, layer, "layer_output", *result);
    return result;
}
#else
static DenseTensor gemma_rms_norm_f16(DenseTensor x,
                                 ::tyloi::DenseTensor weight, double eps,
                                 double weight_offset) {
    TYLOINFER_RUNTIME_CHECK(x.scalar_type() == ::tyloi::kFloat16,
                            "gemma_rms_norm_f16: activation must remain f16");
    return rms_norm_f16_cuda(x.contiguous(), weight, eps, weight_offset);
}

static DenseTensor forward_ffn(const Layer& weights, Executor& executor, const Executor::Mode& mode,
                          DenseTensor residual, DenseTensor oo, int64_t B, int64_t T, int64_t H) {
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
    auto measure = [&](const char* name, auto&& operation) {
        return executor.profiler().measure(name, std::forward<decltype(operation)>(operation));
    };
    trace_stage(mode, layer, "attention_output", oo);
    DenseTensor prepared_residual, dense_input, router_input, moe_input;
    const bool fused_norms = gemma4_moe && mode.gemma4_fused_norms &&
                             mode.trace == nullptr && layer_scale.defined();
    if (fused_norms) {
        auto prepared = measure("gemma.attn_residual_pre_norms", [&]() {
            return gemma4_attn_residual_pre_norms_f16_cuda(
                residual.reshape({B * T, H}), oo.reshape({B * T, H}), attn_post_norm, ffn_norm,
                gemma_router_norm_scale, ffn_pre_norm_2, rms_norm_eps);
        });
        prepared_residual = prepared[0].reshape({B, T, H});
        dense_input = prepared[1];
        router_input = prepared[2];
        moe_input = prepared[3];
    }
    if (!fused_norms) {
        {
            auto attn_post = measure("gemma.attn_post_norm", [&]() {
                return gemma_rms_norm_f16(oo.reshape({B * T, H}), attn_post_norm, rms_norm_eps,
                                          norm_weight_offset);
            });
            prepared_residual = measure("gemma.attn_residual", [&] {
                return acc_cuda(residual.reshape({B * T, H}), attn_post).reshape({B, T, H});
            });
            trace_stage(mode, layer, "attention_residual", prepared_residual);
        }
        dense_input = measure("gemma.ffn_pre_norm", [&] {
            return gemma_rms_norm_f16(prepared_residual.reshape({B * T, H}), ffn_norm, rms_norm_eps,
                                      norm_weight_offset);
        });
        if (gemma4_moe) {
            router_input = measure("gemma.router_norm", [&]() {
                return qwen_rms_norm(prepared_residual.reshape({B * T, H}).to(::tyloi::kFloat32),
                                     gemma_router_norm_scale, rms_norm_eps, norm_weight_offset);
            });
            moe_input = measure("gemma.ffn_pre_norm_2", [&]() {
                return gemma_rms_norm_f16(prepared_residual.reshape({B * T, H}), ffn_pre_norm_2, rms_norm_eps,
                                          norm_weight_offset);
            });
        }
    }
    residual = prepared_residual;
    auto normalize = [&](DenseTensor value, const DenseTensor& weight, const char* name) {
        return measure(name, [&] {
            return value.scalar_type() == ::tyloi::kFloat16
                       ? gemma_rms_norm_f16(value, weight, rms_norm_eps, norm_weight_offset)
                       : qwen_rms_norm(value.to(::tyloi::kFloat32), weight,
                                       rms_norm_eps, norm_weight_offset)
                             .to(::tyloi::kFloat16)
                             .contiguous();
        });
    };
    auto result = measure("gemma.ffn_dense", [&] {
        return ffn.forward(dense_input, executor, mode.ffn).reshape({B * T, H});
    });
    if (gemma4_moe) {
        if (!fused_norms) {
            auto dense = std::move(result);
            result = normalize(dense, ffn_post_norm_1, "gemma.ffn_post_norm_1");
            trace_stage(mode, layer, "dense_output", result);
        }
#if TYLOINFER_ENABLE_MFQ_COMPAT
        DenseTensor moe_output;
        {
            struct SelectedExperts { DenseTensor ids, weights; } selected;
            {
                auto router_logits = measure("gemma.router", [&]() {
                    return ::tyloi::cuda::matmul(router_input, gemma_router.transpose(0, 1));
                });
                selected = measure("gemma.topk", [&]() {
                    auto top = moe_topk_cuda(router_logits.contiguous(), gemma_top_k, false, false,
                                             false, true, mfq_nullopt, 1e-20, 1.0);
                    return SelectedExperts{top.at(0), top.at(1)};
                });
            }
            trace_stage(mode, layer, "route_ids", selected.ids);
            trace_stage(mode, layer, "route_weights_before_scale", selected.weights);
            measure("gemma.route_scale", [&]() {
                return moe_apply_expert_scale_cuda(selected.weights, selected.ids, gemma_expert_scale);
            });
            trace_stage(mode, layer, "route_weights", selected.weights);
            DenseTensor down_pair;
            {
                auto route = measure("gemma.route_map", [&]() {
                    return build_moe_route_plan(selected.ids, gemma_moe_gate_up.n_experts);
                });
                const bool projection_bundle_prefetched = prefetch_cached_moe_projection_bundle(
                    gemma_moe_gate_up, gemma_moe_down, route);
                const bool tracing = mode.trace != nullptr && layer == mode.trace_layer;
                DenseTensor packed;
                if (!tracing && gemma_moe_gate_up.supports_projection_glu_epilogue()) {
                    auto hidden = measure("gemma.moe_gate_up_geglu", [&] {
                        return executor.projector(mode.ffn).glu(gemma_moe_gate_up, moe_input, route, true);
                    });
                    if (!projection_bundle_prefetched)
                        tyloi::cuda::prefetch_experts(gemma_moe_down, route);
                    down_pair = measure("gemma.moe_down", [&] {
                        return executor.projector(mode.ffn).run(gemma_moe_down, hidden, route);
                    });
                } else {
                    packed = measure("gemma.moe_gate_up", [&] {
                        return executor.projector(mode.ffn).run(gemma_moe_gate_up, moe_input, route);
                    });
                    if (!projection_bundle_prefetched)
                        tyloi::cuda::prefetch_experts(gemma_moe_down, route);
                    trace_stage(mode, layer, "moe_gate_up", packed);
                    if (!tracing && packed.size(0) <= 4) {
                        down_pair = measure("gemma.moe_geglu_down", [&] {
                            return executor.projector(mode.ffn).gated(gemma_moe_down, packed, route, true);
                        });
                    } else {
                        auto hidden = measure("gemma.moe_geglu", [&] { return moe_geglu_split_cuda(packed); });
                        if (tracing)
                            trace_stage(mode, layer, "moe_hidden", hidden);
                        down_pair = measure("gemma.moe_down", [&] {
                            return executor.projector(mode.ffn).run(gemma_moe_down, hidden, route);
                        });
                    }
                }
                trace_stage(mode, layer, "moe_down", down_pair);
            }
            moe_output = measure("gemma.moe_reduce", [&]() {
                return moe_weighted_reduce_cuda(down_pair, selected.weights);
            });
            trace_stage(mode, layer, "moe_reduce", moe_output);
        }
        if (fused_norms) {
            result = measure("gemma.ffn_merge", [&] {
                return gemma4_ffn_merge_f16_cuda(result, moe_output, residual.reshape({B * T, H}),
                                                 ffn_post_norm_1, ffn_post_norm_2, ffn_post_norm,
                                                 layer_scale, rms_norm_eps)
                    .reshape({B, T, H});
            });
        } else {
            auto normalized = normalize(std::move(moe_output), ffn_post_norm_2, "gemma.ffn_post_norm_2");
            result = measure("gemma.ffn_combine", [&] {
                return result + normalized;
            });
        }
#else
        throw std::runtime_error("Legacy Gemma MoE weights require a compatibility build");
#endif
    }
    if (!fused_norms) {
        {
            auto normalized = normalize(std::move(result), ffn_post_norm, "gemma.ffn_post_norm");
            result = measure("gemma.ffn_residual", [&] {
                return acc_cuda(residual.reshape({B * T, H}), normalized).reshape({B, T, H});
            });
        }
        if (layer_scale.defined()) {
            result = measure("gemma.layer_scale", [&] {
                return result * layer_scale;
            });
        }
    }
    trace_stage(mode, layer, "layer_output", result);
    return result;
}
#endif

namespace {

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
std::shared_ptr<tyloi::Tensor> normalize_ids(int primary, const tyloi::Tensor& ids) {
    tyloi::models::require_model(ids.dim() == 1 || ids.dim() == 2,
                               "token IDs must have shape [tokens] or [batch,tokens]");
    auto shaped = ids.dim() == 1 ? tyloi::ops::reshape(ids, {1, ids.size(0)})
                                 : tyloi::ops::copy(ids, ids.device());
    tyloi::models::require_model(shaped->size(0) > 0 && shaped->size(1) > 0,
                               "token IDs must not be empty");
    auto converted = tyloi::ops::cast(*shaped, tyloi::ScalarType::int64);
    auto copied = tyloi::ops::copy(*converted, tyloi::Device{tyloi::DeviceType::cuda, primary});
    return tyloi::ops::contiguous(*copied);
}
#else
DenseTensor normalize_ids(Collectives& collectives, int primary, DenseTensor ids) {
    tyloi::models::require_model(ids.dim() == 1 || ids.dim() == 2,
                               "token IDs must have shape [tokens] or [batch,tokens]");
    if (ids.dim() == 1) {
        ids = ids.unsqueeze(0);
    }
    tyloi::models::require_model(ids.size(0) > 0 && ids.size(1) > 0,
                               "token IDs must not be empty");
    return collectives.copy(ids.to(::tyloi::kInt64), primary).contiguous();
}
#endif

} // namespace

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
std::shared_ptr<tyloi::Tensor> Model::embedding(const tyloi::Tensor& ids, int primary) const {
    auto normalized = normalize_ids(primary, ids);
    return tyloi::ops::measure("model.embed", normalized->device(), [&] {
        return tyloi::ops::embedding(*normalized, *embed);
    });
#else
DenseTensor embed(const Model& model, Executor& executor, Collectives& collectives,
             int primary, DenseTensor ids, const Executor::Mode& mode) {
    ids = normalize_ids(collectives, primary, std::move(ids));
    const Task task{task::Embed{0, 0, *model.embed}, -1, {}, "model.embed"};
    Inputs inputs;
    inputs.values = {std::move(ids)};
    return executor.run(std::span<const Task>{&task, 1}, std::move(inputs), mode);
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
std::shared_ptr<tyloi::Tensor> Model::normalize(const tyloi::Tensor& hidden, std::int64_t batch, std::int64_t tokens) const {
    const auto& model = *this;
    const std::array<std::int64_t, 3> shape{batch, tokens, model.config.hidden_size};
    auto shaped = tyloi::ops::reshape(hidden, shape);
    return tyloi::ops::measure("model.output_norm", shaped->device(), [&] {
        return tyloi::ops::rms_norm(*shaped, model.output_norm,
                                       model.config.rms_norm_eps, 0.0);
    });
#else
DenseTensor normalize(const Model& model, Executor& executor, DenseTensor hidden,
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
std::shared_ptr<tyloi::Tensor> Model::softcap(const tyloi::Tensor& logits) const {
    return tyloi::ops::softcap(logits, config.final_logit_softcapping);
#else
DenseTensor softcap(const Model& model, Executor& executor, DenseTensor logits) {
    const Task task{task::Softcap{0, 0, model.config.final_logit_softcapping}};
    Inputs inputs;
    inputs.values = {std::move(logits)};
    return executor.run(std::span<const Task>{&task, 1}, std::move(inputs), {});
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
std::shared_ptr<tyloi::Tensor> Model::logits(const tyloi::Tensor& hidden, bool capped) const {
    const auto& model = *this;
    auto result = tyloi::ops::measure("model.lm_head", hidden.device(), [&] {
        return tyloi::ops::project(hidden, *model.lm_head);
    });
    return capped ? tyloi::ops::softcap(*result, model.config.final_logit_softcapping)
                  : std::move(result);
#else
DenseTensor logits(const Model& model, Executor& executor, DenseTensor hidden, bool capped,
              const Executor::Mode& mode) {
    std::array<Task, 2> tasks{
        Task{task::Project{0, 0, *model.lm_head}, -1, {}, "model.lm_head"},
        Task{task::Softcap{0, 0, model.config.final_logit_softcapping}}};
    Inputs inputs;
    inputs.values = {std::move(hidden)};
    return executor.run(std::span<const Task>{tasks.data(), capped ? tasks.size() : 1},
                        std::move(inputs), mode);
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
std::shared_ptr<tyloi::Tensor> forward_layer(const Layer& layer, tyloi::state::KvView& cache, const tyloi::Tensor& input, const AttentionStep& step,
                     const Model::Options& mode) {
#else
DenseTensor forward_layer(const Layer& layer, KVCache& cache, Executor& executor,
                     std::size_t scratch_index, DenseTensor input, const AttentionStep& step,
                     const Executor::Mode& mode) {
#endif
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
    MfqCudaGuard guard(layer.device);
#endif
    const auto B = input.size(0), T = input.size(1), H = input.size(2);
    const auto& attention = layer.attention;
    const auto capacity = attention.sliding ? attention.attention_window
        : (mode.kv_capacity > 0 ? std::max<std::int64_t>(step.cache_position + T, mode.kv_capacity)
                                : attention.max_position_embeddings);
    if (!cache.defined()) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        cache = tyloi::state::allocate_kv(B, attention.kv_heads, capacity, attention.attention_head_dim,
                        attention.sliding, input.device(),
                        attention.official_bf16 ? tyloi::ScalarType::bfloat16 : tyloi::ScalarType::float16,
                        attention.kv_quantization);
#else
        cache = KVCache(B, attention.kv_heads, capacity, attention.attention_head_dim,
                        attention.sliding, input.device(),
                        attention.official_bf16 ? ::tyloi::kBFloat16
                                                : ::tyloi::kFloat16,
                        attention.kv_quantization);
#endif
    }
    cache.quantization = attention.kv_quantization;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    auto result = attention.forward(input, step,
        cache, attention.attention_rope,
        attention.layer == mode.trace_layer ? mode.trace : nullptr);
#else
    auto result = attention.forward(executor.projector(mode.ffn), executor.profiler(),
        executor.attention_options(), std::move(input), step, cache.view(),
        executor.attention_scratch(scratch_index), attention.attention_rope,
        attention.layer == mode.trace_layer ? mode.trace : nullptr);
#endif
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    return forward_ffn(layer, mode,
                       *result[0], *result[1], B, T, H);
#else
    return forward_ffn(layer, executor, mode,
                       std::move(result[0]), std::move(result[1]), B, T, H);
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
tyloi::StepSequence<tyloi::models::Prepared> Model::prepare(
    const std::vector<std::int64_t>&, const MfqMultimodalInput&, tyloi::models::State&) const {
    throw std::invalid_argument("model has no multimodal component");
}

std::shared_ptr<tyloi::Tensor> Model::forward(tyloi::models::Input input, tyloi::models::State& state_value,
                                              const Options& mode) const {
    auto& state = dynamic_cast<State&>(state_value);
    const auto& model = *this;
    tyloi::models::require_model(input.ids != nullptr,
                               "token IDs must have shape [tokens] or [batch,tokens]");
    auto ids = normalize_ids(mode.primary, *input.ids);
    std::shared_ptr<const tyloi::Tensor> embeddings = input.embeddings
        ? input.embeddings : embedding(*ids, mode.primary);
    const auto B = ids->size(0), T = ids->size(1);
    tyloi::models::require_model(embeddings->dim() == 3 && embeddings->size(0) == B &&
                                   embeddings->size(1) == T &&
                                   embeddings->size(2) == model.config.hidden_size,
                               "inputs_embeds shape must match [batch,tokens,hidden_size]");
    tyloi::models::require_model(input.confirmed >= 0 && input.confirmed == 0,
                               "unsupported speculative backbone geometry");
    if (state.cache_pos == 0) state.reset(B);

    const tyloi::Device primary{tyloi::DeviceType::cuda, mode.primary};
    auto positions_on_primary = [&](const tyloi::Tensor& value) {
        auto converted = tyloi::ops::cast(value, tyloi::ScalarType::int64);
        auto copied = tyloi::ops::copy(*converted, primary);
        return tyloi::ops::contiguous(*copied);
    };
    auto physical = input.cache_positions ? positions_on_primary(*input.cache_positions)
        : tyloi::ops::arange(state.cache_pos, state.cache_pos + T, primary);
    auto semantic = input.positions ? positions_on_primary(*input.positions)
        : (state.decode_position_delta == 0 ? physical
            : tyloi::ops::add(*physical, state.decode_position_delta));
    tyloi::models::require_model((physical->dim() == 1 && physical->numel() == T) ||
                                   (physical->dim() == 2 && physical->size(0) == B && physical->size(1) == T),
                               "cache_positions must have shape [tokens] or [batch,tokens]");
    tyloi::models::require_model((semantic->dim() == 1 && semantic->size(0) == T) ||
                                   (semantic->dim() == 2 && semantic->size(0) == B && semantic->size(1) == T),
                               "position_ids must have shape [tokens] or [batch,tokens]");
    if (input.attention_mask) {
        const auto& mask = *input.attention_mask;
        tyloi::models::require_model(mask.dim() == 2 && mask.size(0) == B &&
                                       mask.size(1) >= state.cache_pos + T,
                                   "attention_mask must cover [batch,cached_plus_current_tokens]");
    }
    if (!input.lengths && state.cache_pos > 0 && T == 1) {
        const std::array<std::int64_t, 1> shape{B};
        input.lengths = tyloi::ops::full(shape, state.cache_pos + 1, primary);
    }
    AttentionStep step{semantic, input.lengths, input.cache_positions ? physical : nullptr,
                       {}, state.cache_pos, input.kv_length, input.attention_parts};
    auto copied = tyloi::ops::copy(*embeddings, primary);
    auto hidden = tyloi::ops::contiguous(*copied);
    hidden = tyloi::ops::measure("model.embed_scale", hidden->device(), [&] {
        return tyloi::ops::scale(*hidden, model.embed_scale);
    });
    const auto trace = [&](const tyloi::Tensor& value) {
        if (input.trace) {
            auto local = tyloi::ops::copy(value, primary);
            auto converted = tyloi::ops::cast(*local, tyloi::ScalarType::float32);
            input.trace->push_back(tyloi::ops::clone(*converted));
        }
    };
    trace(*hidden);
    for (std::size_t index = 0; index < model.layers.size(); ++index) {
        const auto& layer = model.layers[index];
        auto local = step;
        const tyloi::Device device{tyloi::DeviceType::cuda, layer.device};
        local.positions = tyloi::ops::copy(*step.positions, device);
        if (step.sequence_lengths)
            local.sequence_lengths = tyloi::ops::copy(*step.sequence_lengths, device);
        if (step.cache_positions)
            local.cache_positions = tyloi::ops::copy(*step.cache_positions, device);
        auto local_hidden = tyloi::ops::copy(*hidden, device);
        hidden = forward_layer(layer, state.layers[index], *local_hidden, local, mode);
        trace(*hidden);
    }
    if (!input.positions || input.advance) state.cache_pos += T;
    hidden = tyloi::ops::copy(*hidden, primary);
    auto finalized = normalize(*hidden, B, T);
    if (input.raw_hidden) *input.raw_hidden = std::move(hidden);
    return finalized;
}
#else
DenseTensor forward(const Model& model, State& state, Executor& executor, Collectives& collectives,
               const Executor::Mode& mode, Input input) {
    MfqCudaGuard guard(mode.primary);
    auto ids = normalize_ids(collectives, mode.primary, std::move(input.ids));
    auto embeddings = input.embeddings.has_value() ? std::move(*input.embeddings)
        : embed(model, executor, collectives, mode.primary, ids, mode);
    const auto B = ids.size(0), T = ids.size(1);
    tyloi::models::require_model(embeddings.dim() == 3 && embeddings.size(0) == B &&
                                   embeddings.size(1) == T &&
                                   embeddings.size(2) == model.config.hidden_size,
                               "inputs_embeds shape must match [batch,tokens,hidden_size]");
    tyloi::models::require_model(input.confirmed >= 0 && input.confirmed == 0,
                               "unsupported speculative backbone geometry");
    if (state.cache_pos == 0) state.reset(B);

    const auto options = ::tyloi::TensorOptions().device(::tyloi::Device(::tyloi::kCUDA, mode.primary))
                             .dtype(::tyloi::kInt64);
    auto physical = input.cache_positions.has_value()
        ? collectives.copy(input.cache_positions->to(::tyloi::kInt64), mode.primary).contiguous()
        : ::tyloi::cuda::arange(state.cache_pos, state.cache_pos + T, options);
    auto semantic = input.positions.has_value()
        ? collectives.copy(input.positions->to(::tyloi::kInt64), mode.primary).contiguous()
        : (state.decode_position_delta == 0 ? physical : physical + state.decode_position_delta);
    tyloi::models::require_model((physical.dim() == 1 && physical.numel() == T) ||
                                   (physical.dim() == 2 && physical.size(0) == B && physical.size(1) == T),
                               "cache_positions must have shape [tokens] or [batch,tokens]");
    tyloi::models::require_model((semantic.dim() == 1 && semantic.size(0) == T) ||
                                   (semantic.dim() == 2 && semantic.size(0) == B && semantic.size(1) == T),
                               "position_ids must have shape [tokens] or [batch,tokens]");
    if (input.attention_mask.has_value()) {
        const auto& mask = *input.attention_mask;
        tyloi::models::require_model(mask.dim() == 2 && mask.size(0) == B &&
                                       mask.size(1) >= state.cache_pos + T,
                                   "attention_mask must cover [batch,cached_plus_current_tokens]");
    }
    if (!input.lengths.has_value() && state.cache_pos > 0 && T == 1) {
        input.lengths = ::tyloi::cuda::full({B}, state.cache_pos + 1, options);
    }
    AttentionStep step{semantic, input.lengths ? MfqOptional<DenseTensor>(DenseTensor(*input.lengths)) : mfq_nullopt,
                       input.cache_positions.has_value() ? MfqOptional<DenseTensor>(physical) : mfq_nullopt,
                       {}, state.cache_pos, input.kv_length, input.attention_parts};
    auto hidden = collectives.copy(std::move(embeddings), mode.primary).contiguous();
    hidden = executor.profiler().measure("model.embed_scale", [&] {
        return hidden * model.embed_scale;
    });
    const auto trace = [&](const DenseTensor& value) {
        if (input.trace) {
            input.trace->push_back(collectives.copy(value, mode.primary).to(::tyloi::kFloat32).clone());
        }
    };
    trace(hidden);
    for (std::size_t index = 0; index < model.layers.size(); ++index) {
        const auto& layer = model.layers[index];
        {
            MfqCudaGuard layer_guard(layer.device);
            auto local = step;
            local.positions = collectives.copy(step.positions, layer.device);
            if (step.sequence_lengths.has_value())
                local.sequence_lengths = collectives.copy(*step.sequence_lengths, layer.device);
            if (step.cache_positions.has_value())
                local.cache_positions = collectives.copy(*step.cache_positions, layer.device);
            hidden = forward_layer(layer, state.layers[index], executor, index,
                                   collectives.copy(std::move(hidden), layer.device), local, mode);
        }
        trace(hidden);
    }
    if (!input.positions.has_value() || input.advance) state.cache_pos += T;
    hidden = collectives.copy(std::move(hidden), mode.primary);
    auto finalized = normalize(model, executor, hidden, B, T);
    if (input.raw_hidden) *input.raw_hidden = std::move(hidden);
    return finalized;
}
#endif

} // namespace tyloi::cuda::gemma4
