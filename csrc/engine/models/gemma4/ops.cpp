#include "engine/models/common/load.h"
#include "engine/weights/moe/experts.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/backends/cuda/kernels/mfq_cuda_moe_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/weights/loader.h"
#include "engine/models/gemma4/ffn.h"
#include "engine/models/gemma4/forward.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/models/gemma4/load.h"
#include "engine/models/gemma4/state.h"
#include "engine/models/gemma4/weights.h"
#include "engine/backends/cuda/executor/options.h"

#include "engine/ops.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <span>

namespace mfq::cuda::gemma4 {

static void trace_stage(const Executor::Mode& mode, int layer, const char* name,
                        const Tensor& value) {
    if (mode.trace != nullptr && layer == mode.trace_layer)
        mode.trace->emplace_back(name, value.to(mfq_tensor_backend::kFloat32).contiguous().clone());
}

static mfq_tensor_backend::Tensor gemma_rms_norm_f16(mfq_tensor_backend::Tensor x,
                                                     mfq_tensor_backend::Tensor weight, double eps,
                                                     double weight_offset) {
    TYLOINFER_RUNTIME_CHECK(x.scalar_type() == mfq_tensor_backend::kFloat16,
                      "gemma_rms_norm_f16: activation must remain f16");
    return rms_norm_f16_cuda(x.contiguous(), weight, eps, weight_offset);
}

static Tensor forward_ffn(const Layer& weights, Executor& executor, const Executor::Mode& mode,
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
    auto &profiler = executor.profiler();
    trace_stage(mode, layer, "attention_output", oo);
    const bool fused_norms = gemma4_moe && mode.gemma4_fused_norms &&
                             mode.trace == nullptr && layer_scale.defined();
    using Tensor = mfq_tensor_backend::Tensor;
    using Inputs = mfq::models::gemma4::FfnInputs<Tensor>;
    auto inputs = mfq::models::gemma4::prepare_ffn(
        residual, oo, gemma4_moe,
        [&](Tensor residual, Tensor oo) -> std::optional<Inputs> {
            if (!fused_norms)
                return {};
            auto prepared = profiler.measure("gemma.attn_residual_pre_norms", [&]() {
                return gemma4_attn_residual_pre_norms_f16_cuda(
                    residual.reshape({B * T, H}), oo.reshape({B * T, H}), attn_post_norm, ffn_norm,
                    gemma_router_norm_scale, ffn_pre_norm_2, rms_norm_eps);
            });
            return Inputs{prepared[0].reshape({B, T, H}), prepared[1], prepared[2], prepared[3]};
        },
        [&](Tensor oo) {
            return profiler.measure("gemma.attn_post_norm", [&]() {
                return gemma_rms_norm_f16(oo.reshape({B * T, H}), attn_post_norm, rms_norm_eps,
                                          norm_weight_offset);
            });
        },
        [&](Tensor residual, Tensor attn_post) {
            auto x = profiler.measure("gemma.attn_residual", [&] {
                return acc_cuda(residual.reshape({B * T, H}), attn_post).reshape({B, T, H});
            });
            trace_stage(mode, layer, "attention_residual", x);
            return x;
        },
        [&](Tensor x) {
            return profiler.measure("gemma.ffn_pre_norm", [&] {
                return gemma_rms_norm_f16(x.reshape({B * T, H}), ffn_norm, rms_norm_eps,
                                          norm_weight_offset);
            });
        },
        [&](Tensor x) {
            return profiler.measure("gemma.router_norm", [&]() {
                return qwen_rms_norm(x.reshape({B * T, H}).to(mfq_tensor_backend::kFloat32),
                                     gemma_router_norm_scale, rms_norm_eps, norm_weight_offset);
            });
        },
        [&](Tensor x) {
            return profiler.measure("gemma.ffn_pre_norm_2", [&]() {
                return gemma_rms_norm_f16(x.reshape({B * T, H}), ffn_pre_norm_2, rms_norm_eps,
                                          norm_weight_offset);
            });
        });
    residual = inputs.residual;
    auto dense_input = inputs.dense, router_input = inputs.router, moe_input = inputs.experts;
    auto result = mfq::models::gemma4::feed_forward(
        gemma4_moe, layer_scale.defined(), fused_norms, residual,
        [&] {
            return profiler.measure("gemma.ffn_dense", [&] {
                return ffn.forward(dense_input, executor, mode.ffn).reshape({B * T, H});
            });
        },
        [&] {
#if TYLOINFER_ENABLE_MFQ_COMPAT
            return mfq::models::gemma4::experts(
                [&] {
                    return profiler.measure("gemma.router", [&]() {
                        return mfq_tensor_backend::matmul(router_input,
                                                          gemma_router.transpose(0, 1));
                    });
                },
                [&](Tensor router_logits) {
                    return profiler.measure("gemma.topk", [&]() {
                        return moe_topk_cuda(router_logits.contiguous(), gemma_top_k, false, false,
                                             false, true, mfq_nullopt, 1e-20, 1.0);
                    });
                },
                [&](auto &selected) {
                    trace_stage(mode, layer, "route_ids", selected.at(0));
                    trace_stage(mode, layer, "route_weights_before_scale",
                                      selected.at(1));
                    profiler.measure("gemma.route_scale", [&]() {
                        return moe_apply_expert_scale_cuda(selected.at(1), selected.at(0),
                                                           gemma_expert_scale);
                    });
                    trace_stage(mode, layer, "route_weights", selected.at(1));
                },
                [&](const auto &selected) {
                    auto route = profiler.measure("gemma.route_map", [&]() {
                        return build_moe_route_plan(selected.at(0), gemma_moe_gate_up.n_experts);
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
                            auto hidden = profiler.measure("gemma.moe_gate_up_geglu", [&] {
                                return executor.projector(mode.ffn).glu(gemma_moe_gate_up, input, route, true);
                            });
                            prefetch_down();
                            return hidden;
                        },
                        [&](Tensor input) {
                            packed = profiler.measure("gemma.moe_gate_up", [&] {
                                return executor.projector(mode.ffn).run(gemma_moe_gate_up, input, route);
                            });
                            prefetch_down();
                            trace_stage(mode, layer, "moe_gate_up", packed);
                            const auto width = packed.size(-1) / 2;
                            return std::array<Tensor, 2>{packed.narrow(-1, 0, width),
                                                         packed.narrow(-1, width, width)};
                        },
                        [&](const Tensor &, const Tensor &, auto, double) {
                            auto hidden = profiler.measure(
                                "gemma.moe_geglu", [&] { return moe_geglu_split_cuda(packed); });
                            if (tracing)
                                trace_stage(mode, layer, "moe_hidden", hidden);
                            return hidden;
                        },
                        [&](Tensor hidden) {
                            return profiler.measure("gemma.moe_down", [&] {
                                return executor.projector(mode.ffn).run(gemma_moe_down, hidden, route);
                            });
                        },
                        [&](const Tensor &, const Tensor &, auto, double) -> std::optional<Tensor> {
                            if (tracing || packed.size(0) > 4)
                                return {};
                            return profiler.measure("gemma.moe_geglu_down", [&] {
                                return executor.projector(mode.ffn).gated(gemma_moe_down, packed, route, true);
                            });
                        });
                    trace_stage(mode, layer, "moe_down", down_pair);
                    return down_pair;
                },
                [&](Tensor down_pair, const auto &selected) {
                    auto moe_output = profiler.measure("gemma.moe_reduce", [&]() {
                        return moe_weighted_reduce_cuda(down_pair, selected.at(1));
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
            auto normalized = profiler.measure(name, [&] {
                return value.scalar_type() == mfq_tensor_backend::kFloat16
                           ? gemma_rms_norm_f16(value, weight, rms_norm_eps, norm_weight_offset)
                           : qwen_rms_norm(value.to(mfq_tensor_backend::kFloat32), weight,
                                           rms_norm_eps, norm_weight_offset)
                                 .to(mfq_tensor_backend::kFloat16)
                                 .contiguous();
            });
            if (role == Norm::dense)
                trace_stage(mode, layer, "dense_output", normalized);
            return normalized;
        },
        [&](Tensor dense, Tensor routed) {
            return profiler.measure("gemma.ffn_combine", [&] { return dense + routed; });
        },
        [&](Tensor skip, Tensor value) {
            return profiler.measure("gemma.ffn_residual", [&] {
                return acc_cuda(skip.reshape({B * T, H}), value).reshape({B, T, H});
            });
        },
        [&](Tensor value) {
            return profiler.measure("gemma.layer_scale", [&] { return value * layer_scale; });
        },
        [&](Tensor dense, Tensor routed, Tensor skip) {
            return profiler.measure("gemma.ffn_merge", [&] {
                return gemma4_ffn_merge_f16_cuda(dense, routed, skip.reshape({B * T, H}),
                                                 ffn_post_norm_1, ffn_post_norm_2, ffn_post_norm,
                                                 layer_scale, rms_norm_eps)
                    .reshape({B, T, H});
            });
        });
    trace_stage(mode, layer, "layer_output", result);
    return result;
}

namespace {

Tensor normalize_ids(Collectives& collectives, int primary, Tensor ids) {
    mfq::models::require_model(ids.dim() == 1 || ids.dim() == 2,
                               "token IDs must have shape [tokens] or [batch,tokens]");
    if (ids.dim() == 1) ids = ids.unsqueeze(0);
    mfq::models::require_model(ids.size(0) > 0 && ids.size(1) > 0,
                               "token IDs must not be empty");
    return collectives.copy(ids.to(mfq_tensor_backend::kInt64), primary).contiguous();
}

} // namespace

Tensor embed(const Model& model, Executor& executor, Collectives& collectives,
             int primary, Tensor ids, const Executor::Mode& mode) {
    const Task task{task::Embed{0, 0, model.embed}, -1, {}, "model.embed"};
    Inputs inputs;
    inputs.values = {normalize_ids(collectives, primary, std::move(ids))};
    return executor.run(std::span<const Task>{&task, 1}, std::move(inputs), mode);
}

Tensor normalize(const Model& model, Executor& executor, Tensor hidden,
                 std::int64_t batch, std::int64_t tokens) {
    hidden = hidden.reshape({batch, tokens, model.metadata.hidden_size});
    const Task task{task::Norm{0, 0, model.output_norm, model.metadata.rms_norm_eps,
                               model.metadata.norm_weight_offset},
                    -1, {}, "model.output_norm"};
    Inputs inputs;
    inputs.values = {std::move(hidden)};
    return executor.run(std::span<const Task>{&task, 1}, std::move(inputs), {});
}

Tensor softcap(const Model& model, Executor& executor, Tensor logits) {
    const Task task{task::Softcap{0, 0, model.metadata.final_logit_softcapping}};
    Inputs inputs;
    inputs.values = {std::move(logits)};
    return executor.run(std::span<const Task>{&task, 1}, std::move(inputs), {});
}

Tensor logits(const Model& model, Executor& executor, Tensor hidden, bool capped,
              const Executor::Mode& mode) {
    std::array<Task, 2> tasks{
        Task{task::Project{0, 0, model.lm_head}, -1, {}, "model.lm_head"},
        Task{task::Softcap{0, 0, model.metadata.final_logit_softcapping}}};
    Inputs inputs;
    inputs.values = {std::move(hidden)};
    return executor.run(std::span<const Task>{tasks.data(), capped ? tasks.size() : 1},
                        std::move(inputs), mode);
}

Tensor forward_layer(const Layer& layer, KVCache& cache, Executor& executor,
                     std::size_t scratch_index, Tensor input, const AttentionStep& step,
                     const Executor::Mode& mode) {
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
    auto result = attention.forward(executor.projector(mode.ffn), executor.profiler(),
        executor.attention_options(), std::move(input), step, cache.view(),
        executor.attention_scratch(scratch_index), attention.attention_rope,
        attention.layer == mode.trace_layer ? mode.trace : nullptr);
    return forward_ffn(layer, executor, mode, std::move(result[0]), std::move(result[1]), B, T, H);
}

Tensor forward(const Model& model, State& state, Executor& executor, Collectives& collectives,
               const Executor::Mode& mode, Input input) {
    namespace tb = mfq_tensor_backend;
    MfqCudaGuard guard(mode.primary);
    auto ids = normalize_ids(collectives, mode.primary, std::move(input.ids));
    auto embeddings = input.embeddings.has_value() ? std::move(*input.embeddings)
        : embed(model, executor, collectives, mode.primary, ids, mode);
    const auto B = ids.size(0), T = ids.size(1);
    mfq::models::require_model(embeddings.dim() == 3 && embeddings.size(0) == B &&
                                   embeddings.size(1) == T &&
                                   embeddings.size(2) == model.metadata.hidden_size,
                               "inputs_embeds shape must match [batch,tokens,hidden_size]");
    mfq::models::require_model(input.confirmed >= 0 && input.confirmed == 0,
                               "unsupported speculative backbone geometry");
    if (state.cache_pos == 0) state.reset(B);

    const auto options = tb::TensorOptions().device(tb::Device(tb::kCUDA, mode.primary))
                             .dtype(tb::kInt64);
    auto physical = input.cache_positions.has_value()
        ? collectives.copy(input.cache_positions->to(tb::kInt64), mode.primary).contiguous()
        : tb::arange(state.cache_pos, state.cache_pos + T, options);
    auto semantic = input.positions.has_value()
        ? collectives.copy(input.positions->to(tb::kInt64), mode.primary).contiguous()
        : (state.decode_position_delta == 0 ? physical : physical + state.decode_position_delta);
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
    if (!input.lengths.has_value() && state.cache_pos > 0 && T == 1)
        input.lengths = tb::full({B}, state.cache_pos + 1, options);
    AttentionStep step{semantic, input.lengths,
                       input.cache_positions.has_value() ? MfqOptional<Tensor>(physical) : mfq_nullopt,
                       {}, state.cache_pos, input.kv_length, input.attention_parts};
    auto hidden = collectives.copy(std::move(embeddings), mode.primary).contiguous();
    hidden = executor.profiler().measure("model.embed_scale", [&] { return hidden * model.embed_scale; });
    const auto trace = [&](const Tensor& value) {
        if (input.trace)
            input.trace->push_back(collectives.copy(value, mode.primary).to(tb::kFloat32).clone());
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

struct LayerLoader : TransformerWeightLoader {
    auto projections(const std::vector<std::string> &names) {
        return weights.group(names, 2);
    }
    auto rope(int64_t capacity, const mfq::models::gemma4::LayerSpec &spec) {
        return RopeCache(capacity, spec.head_dim, spec.rope_base, spec.head_dim, spec.rotary_pairs);
    }
    auto ones(int64_t width) {
        return mfq_tensor_backend::ones({width}, mfq_tensor_backend::TensorOptions()
                                                     .device(mfq_tensor_backend::kCUDA)
                                                     .dtype(mfq_tensor_backend::kFloat32));
    }
    auto layer_scale(const std::string &name) {
        return dense(name).to(mfq_tensor_backend::kFloat16).contiguous();
    }
    auto router_parameter(const std::string &name) {
        return dense(name).to(mfq_tensor_backend::kFloat32).contiguous();
    }
    auto normalized_router_parameter(const std::string &name, double divisor) {
        return (router_parameter(name) / divisor).contiguous();
    }
    auto experts(const std::string &name, int layer, const char *projection) {
        return weights.experts(name, true, layer, projection);
    }
    static auto expert_shape(const mfq::cuda::Experts &weight) {
        return std::array<int64_t, 3>{weight.n_experts, weight.out_per_expert, weight.neuron_len};
    }
};

Layer load_layer(const weight_loader::Loader& weights, const Config &config,
                 int layer, const std::string &type) {
    MfqCudaGuard guard(weights.placement.device);
    Layer result;
    result.device = weights.placement.device;
    LayerLoader loader{{weights}};
    mfq::models::gemma4::load_layer(result.attention, result, loader, config, layer, type);
    return result;
}

Model read(std::shared_ptr<const mfq::ModelSource> source, std::string_view definition,
           std::int64_t context) {
    validate_model_source(*source);
    Model model;
    model.source = std::move(source);
    model.graph = mfq::models::graph(*model.source);
    model.plan = cuda_model_plan(model.graph);
    mfq::models::require_model(model.graph.backbone == "gemma4",
                               "model backbone does not match the requested causal model");
    mfq::models::gemma4::load_config(definition, model.config, model.metadata, model.embed_scale);
    mfq::models::require_model(model.config.num_hidden_layers == model.graph.topology.text_layers,
                               "model graph/config text-layer topology mismatch");
    if (context > 0) {
        mfq::models::require_model(context <= model.metadata.max_position_embeddings,
                                   "--ctx-size exceeds max_position_embeddings");
        model.config.max_position_embeddings = context;
        model.metadata.max_position_embeddings = context;
    }
    return model;
}

void load(Model& model, const weight_loader::Loader& weights, LayerPlacementConfig& placement,
          int gpu_layers, bool expert_parallel, bool load_layers, bool load_input_output) {
    const auto& config = model.config;
    if (expert_parallel && config.num_experts <= 0)
        throw std::runtime_error("--expert-parallel requires a model with routed experts");
    placement.prepare(config.num_hidden_layers);
    const int cpu_layers = gpu_layers < 0 ? 0 :
        static_cast<int>(std::max<int64_t>(config.num_hidden_layers - gpu_layers, 0));
    if (cpu_layers > 0) {
        if (weights.placement.tensor.enabled() || expert_parallel || placement.enabled())
            throw std::runtime_error(
                "--n-gpu-layers cannot be combined with tensor/expert/layer parallelism");
        throw std::runtime_error("--n-gpu-layers currently supports dense Qwen-style blocks only");
    }
    MfqCudaGuard guard(placement.primary_device());
    if (load_input_output) {
        const bool separate = !config.tie_word_embeddings && weights.has("model.output.weight");
        model.embed = weights.embedding("model.token_embedding.weight", separate);
        model.output_norm = weights.tensor("model.output_norm.weight");
        model.lm_head = separate ? weights.projection("model.output.weight", TensorParallelAxis::Output)
            : weights.placement.tensor.enabled()
                ? weights.projection("model.token_embedding.weight", TensorParallelAxis::Output)
                : model.embed;
        if (load_layers) {
            model.layers.reserve(static_cast<std::size_t>(config.num_hidden_layers));
            for (int index = 0; index < config.num_hidden_layers; ++index) {
                const int device = placement.device_for_layer(index);
                MfqCudaGuard layer_guard(device);
                const auto& type = model.metadata.layer_types.at(static_cast<std::size_t>(index));
                std::cerr << "loading layer " << index << ' ' << type << " CUDA" << std::endl;
                model.layers.push_back(load_layer(weights.for_layer(device, false), config, index, type));
            }
        }
    } else {
        if (load_layers) throw std::invalid_argument("streamed model setup requires deferred blocks");
        model.output_norm = weights.tensor("model.output_norm.weight");
    }
}

State::State(const Model& model) : layers(model.layers.size()) {
    layout_.reserve(model.layers.size());
    for (const auto& layer : model.layers) {
        const auto& attention = layer.attention;
        layout_.push_back({layer.device, attention.kv_heads, attention.attention_head_dim,
                           attention.sliding ? attention.attention_window : attention.max_position_embeddings,
                           attention.sliding,
                           attention.official_bf16 ? mfq_tensor_backend::kBFloat16 : mfq_tensor_backend::kFloat16,
                           attention.kv_quantization});
    }
}

void State::preallocate() {
    if (!runtime_options::kv_vram_preallocated()) return;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        if (layers[index].defined()) continue;
        const auto& layout = layout_[index];
        MfqCudaGuard guard(layout.device);
        layers[index] = KVCache(1, layout.heads, layout.capacity, layout.head_dim, layout.sliding,
            mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, layout.device),
            layout.dtype, layout.quantization);
    }
}

void State::reset(std::int64_t next_batch) {
    cache_pos = decode_position_delta = 0;
    batch = next_batch;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        auto& cache = layers[index];
        if (!cache.defined() || cache.batch_size() != next_batch) {
            MfqCudaGuard guard(layout_[index].device);
            cache = KVCache();
        }
    }
}

std::vector<Tensor> State::storage() const {
    std::vector<Tensor> result;
    for (const auto& cache : layers) {
        if (cache.is_paged()) {
            if (cache.k_chunk_ptrs.defined()) result.push_back(cache.k_chunk_ptrs);
            if (cache.v_chunk_ptrs.defined()) result.push_back(cache.v_chunk_ptrs);
            if (cache.page_table.defined()) result.push_back(cache.page_table);
        } else {
            if (cache.k.defined()) result.push_back(cache.k);
            if (cache.v.defined()) result.push_back(cache.v);
        }
    }
    return result;
}

TextSessionStateKind State::text_session_state_kind() const { return TextSessionStateKind::Unsupported; }
bool State::supports_text_session_state() const { return false; }
bool State::supports_paged_text_session_state() const { return false; }
TextSessionState State::capture_text_session_state(const std::vector<std::int64_t>&) const {
    throw std::runtime_error("text session state is unsupported by this model");
}
void State::restore_text_session_state(const TextSessionState&) {
    throw CudaSessionStateError("text session state is unsupported by this model");
}

} // namespace mfq::cuda::gemma4
