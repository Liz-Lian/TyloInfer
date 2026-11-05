#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/native/scope.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/models/session.h"
#include "engine/models/common/check.h"
#include "engine/models/common/load.h"
#include "engine/models/common/model_graph.h"
#include "engine/weights/loader.h"
#include "engine/weights/group.h"
#include "engine/backends/cuda/executor/tq.h"
#include "engine/models/deepseek_v4/load.h"
#include "engine/models/deepseek_v4/forward.h"
#include "engine/backends/cuda/kernels/mfq_cuda_attention_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_cache_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_quant_ops.h"
#include "engine/backends/cuda/kernels/deepseek_v4_attention.h"
#include "engine/backends/cuda/kernels/deepseek_v4_hc.h"
#include "engine/models/common/transformer_layer.h"
#include <iomanip>
#include <limits>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <span>
#include "engine/models/deepseek_v4/snapshot.h"
#include "engine/models/session_codec.h"

namespace tyloi::cuda::deepseek_v4 {

Layer load_layer(const weight_loader::Loader& weights, const Config &c, int i,
                 const std::string &type, Collectives& collectives, int primary,
                 const std::unordered_set<int>& offload, int64_t& host_bytes) {
    const auto &config = c;
    if (type != "deepseek_v4")
        throw std::runtime_error("invalid DeepSeek V4 layer type");
    MfqCudaGuard guard(weights.placement.device);
    const std::string p = "model.block." + std::to_string(i) + ".";
    Layer layer;
    layer.device = weights.placement.device;
    layer.layer = i;
    layer.max_positions = c.max_position_embeddings;
    layer.compress_ratio = config.compress_ratios.at(static_cast<size_t>(i));
    layer.hidden_size = c.hidden_size;
    layer.heads = c.num_attention_heads;
    layer.head_dim = c.head_dim;
    layer.groups = c.o_groups;
    layer.hc_mult = c.hc_mult;
    layer.hc_iterations = c.hc_sinkhorn_iters;
    layer.eps = c.rms_norm_eps;
    layer.hc_eps = c.hc_eps;

    layer.attn_norm = weights.tensor(p + "attention.norm.weight");
    layer.ffn_norm = weights.tensor(p + "mlp.norm.weight");
    layer.q_a_norm = weights.tensor(p + "attention.query_a_norm.weight");
    layer.kv_norm = weights.tensor(p + "attention.key_value_a_norm.weight");
    layer.sinks = weights.tensor(p + "attention.sink")
                   .to(::tyloi::kFloat32)
                   .contiguous();
    layer.hc_attn_fn = weights.tensor(p + "attention.mhc.pre.function")
                        .to(::tyloi::kFloat32)
                        .contiguous();
    layer.hc_attn_scale = weights.tensor(p + "attention.mhc.pre.scale")
                           .to(::tyloi::kFloat32)
                           .contiguous();
    layer.hc_attn_base = weights.tensor(p + "attention.mhc.pre.base")
                          .to(::tyloi::kFloat32)
                          .contiguous();
    layer.hc_ffn_fn = weights.tensor(p + "mlp.mhc.pre.function")
                       .to(::tyloi::kFloat32)
                       .contiguous();
    layer.hc_ffn_scale = weights.tensor(p + "mlp.mhc.pre.scale")
                          .to(::tyloi::kFloat32)
                          .contiguous();
    layer.hc_ffn_base = weights.tensor(p + "mlp.mhc.pre.base")
                         .to(::tyloi::kFloat32)
                         .contiguous();
    layer.q_a = weights.weight(p + "attention.query_a.weight");
    layer.q_b = weights.weight(p + "attention.query_b.weight");
    layer.kv = weights.weight(p + "attention.key_value_a.weight");
    layer.output_a = weights.weight(p + "attention.output_a.weight");
    layer.output_b = weights.weight(p + "attention.output_b.weight");
    layer.attention_rope =
        TailRope(c.max_position_embeddings, c.rope_base, layer.compress_ratio,
                      config.compress_rope_base, config.rope_original_positions, config.rope_factor,
                      config.rope_beta_fast, config.rope_beta_slow);

    if (layer.compress_ratio > 0) {
        layer.compressor.ratio = layer.compress_ratio;
        layer.compressor.head_dim = c.head_dim;
        layer.compressor.overlap = layer.compress_ratio == 4;
        layer.compressor.cache_quant_mode = 1;
        layer.compressor.projection = make_fp32_quant_group(
            collectives, primary,
            weights.group({p + "attention.compressor.key_value.weight",
                                         p + "attention.compressor.gate.weight"}));
        layer.compressor.ape = weights.tensor(p + "attention.compressor.position")
                                .to(::tyloi::kFloat32)
                                .contiguous();
        layer.compressor.norm = weights.tensor(p + "attention.compressor.norm.weight")
                                 .to(::tyloi::kFloat32)
                                 .contiguous();
    }
    if (layer.compress_ratio == 4) {
        layer.indexer_compressor.ratio = 4;
        layer.indexer_compressor.head_dim = c.index_head_dim;
        layer.indexer_compressor.overlap = true;
        layer.indexer_compressor.cache_quant_mode = 2;
        layer.indexer_compressor.projection = make_fp32_quant_group(
            collectives, primary,
            weights.group({p + "attention.indexer.compressor.key_value.weight",
                                         p + "attention.indexer.compressor.gate.weight"}));
        layer.indexer_compressor.ape =
            weights.tensor(p + "attention.indexer.compressor.position")
                .to(::tyloi::kFloat32)
                .contiguous();
        layer.indexer_compressor.norm =
            weights.tensor(p + "attention.indexer.compressor.norm.weight")
                .to(::tyloi::kFloat32)
                .contiguous();
        layer.indexer_q = weights.weight(p + "attention.indexer.query.weight");
        layer.indexer_weight = weights.tensor(p + "attention.indexer.score.weight")
                                .to(::tyloi::kFloat32)
                                .contiguous();
    }

    const bool cpu_offload = offload.count(i) != 0;
    layer.ffn = load_moe_weights(
        weights, p + "mlp.",
        {.layer = i, .cpu_offloaded = cpu_offload, .shared_gate_up_compatible_prefix = 0});
    if (cpu_offload) {
        const int64_t gate_up_bytes =
            layer.ffn.moe_split_gate_up
                ? layer.ffn.moe_gate->backing_nbytes() + layer.ffn.moe_up->backing_nbytes()
                : layer.ffn.moe_gate_up->backing_nbytes();
        const int64_t down_bytes = layer.ffn.moe_down->backing_nbytes();
        host_bytes += gate_up_bytes + down_bytes;
        std::cerr << "cpu_offload layer=" << i << " gate_up_bytes=" << gate_up_bytes
                  << " down_bytes=" << down_bytes
                  << " total_host_bytes=" << host_bytes << std::endl;
    }
    if (i < config.hash_layer_count) {
        layer.ffn.moe_hash_ids = std::make_shared<::tyloi::DenseTensor>(weights.tensor(p + "mlp.router.token_to_expert")
                                  .to(::tyloi::kInt32)
                                  .contiguous());
    }
    layer.ffn.moe_top_k = static_cast<int>(c.num_experts_per_tok);
    layer.ffn.moe_use_sqrt_softplus = true;
    layer.ffn.moe_normalize = c.norm_topk_prob;
    layer.ffn.moe_delayed_softmax = false;
    layer.ffn.moe_shared_ungated = true;
    layer.ffn.moe_router_scale = c.routed_scaling_factor;
    layer.ffn.swiglu_limit = c.swiglu_limit;
    layer.ffn.shared->swiglu_limit = c.swiglu_limit;

    const bool base_shapes =
        layer.attn_norm.numel() == c.hidden_size && layer.ffn_norm.numel() == c.hidden_size &&
        layer.q_a_norm.numel() == c.q_lora_rank && layer.kv_norm.numel() == c.head_dim &&
        layer.sinks.numel() == c.num_attention_heads && layer.q_a->size(1) == c.hidden_size &&
        layer.q_a->size(0) == c.q_lora_rank && layer.q_b->size(1) == c.q_lora_rank &&
        layer.q_b->size(0) == c.num_attention_heads * c.head_dim && layer.kv->size(1) == c.hidden_size &&
        layer.kv->size(0) == c.head_dim &&
        layer.output_a->size(1) == c.num_attention_heads * c.head_dim / c.o_groups &&
        layer.output_a->size(0) == c.o_groups * c.o_lora_rank &&
        layer.output_b->size(1) == c.o_groups * c.o_lora_rank &&
        layer.output_b->size(0) == c.hidden_size;
    const bool gate_up_shapes =
        layer.ffn.moe_split_gate_up
            ? layer.ffn.moe_gate && layer.ffn.moe_up && layer.ffn.moe_gate->size(0) == c.num_experts &&
                  layer.ffn.moe_up->size(0) == c.num_experts &&
                  layer.ffn.moe_gate->size(2) == c.hidden_size &&
                  layer.ffn.moe_up->size(2) == c.hidden_size &&
                  layer.ffn.moe_gate->size(1) == c.moe_intermediate_size &&
                  layer.ffn.moe_up->size(1) == c.moe_intermediate_size
            : layer.ffn.moe_gate_up && layer.ffn.moe_gate_up->size(0) == c.num_experts &&
                  layer.ffn.moe_gate_up->size(2) == c.hidden_size &&
                  layer.ffn.moe_gate_up->size(1) == 2 * c.moe_intermediate_size;
    const bool moe_shapes = gate_up_shapes && layer.ffn.moe_down && layer.ffn.moe_down->size(0) == c.num_experts &&
                            layer.ffn.moe_down->size(2) == c.moe_intermediate_size &&
                            layer.ffn.moe_down->size(1) == c.hidden_size &&
                            layer.ffn.moe_router && layer.ffn.moe_router->size(0) == c.num_experts &&
                            layer.ffn.moe_router->size(1) == c.hidden_size;
    if (!base_shapes || !moe_shapes) {
        throw std::runtime_error("DeepSeek V4 tensor shapes disagree with config at layer " +
                                 std::to_string(i));
    }
    return layer;
}

void validate_offload(const Config &config, const std::unordered_set<int>& offload) {
    if (config.hidden_size != 4096 || config.num_attention_heads != 64 || config.head_dim != 512 ||
        config.q_lora_rank != 1024 || config.qk_rope_head_dim != 64 ||
        config.index_head_dim != 128 || config.index_n_heads != 64 || config.index_topk != 512 ||
        config.o_groups != 8 || config.o_lora_rank != 1024 || config.hc_mult != 4 ||
        config.hc_sinkhorn_iters != 20 || config.num_experts != 256 ||
        config.num_experts_per_tok != 6 || config.moe_intermediate_size != 2048 ||
        config.shared_expert_count != 1 || config.scoring_func != "sqrtsoftplus") {
        throw std::runtime_error("unsupported DeepSeek V4 CUDA configuration");
    }
    if (offload.empty())
        return;
    for (int layer : offload) {
        if (layer < 0 || layer >= config.num_hidden_layers) {
            throw std::runtime_error("CPU-offload layer is outside the model: " +
                                     std::to_string(layer));
        }
    }
}

OutputHeadWeights load_output_head(const weight_loader::Loader& weights) {
    return {
        weights.tensor("model.mhc.output.function")
            .to(::tyloi::kFloat32)
            .contiguous(),
        weights.tensor("model.mhc.output.scale")
            .to(::tyloi::kFloat32)
            .contiguous(),
        weights.tensor("model.mhc.output.base")
            .to(::tyloi::kFloat32)
            .contiguous(),
    };
}

::tyloi::DenseTensor finalize_hidden(::tyloi::DenseTensor hidden,
                                           const OutputHeadWeights &output_head,
                                           const Config &config, CudaProfiler &profiler,
                                           int64_t batch, int64_t tokens) {
    return profiler.measure("model.dsv4_hc_head", [&]() {
        auto flat = hidden.flatten(2).to(::tyloi::kFloat32);
        auto inverse_rms =
            ::tyloi::cuda::rsqrt(flat.square().mean(-1, true) + config.rms_norm_eps);
        auto mixes =
            ::tyloi::cuda::matmul(flat, output_head.function.transpose(0, 1)) * inverse_rms;
        auto pre = ::tyloi::cuda::sigmoid(mixes * output_head.scale + output_head.base) +
                   config.hc_eps;
        return (pre.unsqueeze(-1) *
                flat.reshape({batch, tokens, config.hc_mult, config.hidden_size}))
            .sum(2)
            .to(::tyloi::kFloat16)
            .contiguous();
    });
}

Model read(std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
           int64_t context) {
    validate_model_source(*source);
    Model model;
    model.source = std::move(source);
    model.graph = tyloi::models::graph(*model.source);
    model.plan = cuda_model_plan(model.graph);
    tyloi::models::require_model(model.graph.backbone == "deepseek_v4",
                               "model backbone does not match the requested causal model");
    auto& c = model.config;
    auto& m = model.metadata;
    c = Config::from_json(definition);
    c.layer_types.assign(static_cast<std::size_t>(c.num_hidden_layers), "deepseek_v4");
    m.vocab_size = c.vocab_size;
    m.hidden_size = c.hidden_size;
    m.num_hidden_layers = c.num_hidden_layers;
    m.num_attention_heads = c.num_attention_heads;
    m.num_key_value_heads = c.num_key_value_heads;
    m.head_dim = c.head_dim;
    m.max_position_embeddings = c.max_position_embeddings;
    m.rotary_dim = c.rotary_dim;
    m.num_experts = c.num_experts;
    m.hc_mult = c.hc_mult;
    m.rope_base = c.rope_base;
    m.rms_norm_eps = c.rms_norm_eps;
    m.hc_eps = c.hc_eps;
    m.tie_word_embeddings = c.tie_word_embeddings;
    m.model_type = c.model_type;
    m.layer_types = c.layer_types;
    tyloi::models::require_model(c.num_hidden_layers == model.graph.topology.text_layers,
                               "model graph/config text-layer topology mismatch");
    if (context > 0) {
        tyloi::models::require_model(context <= m.max_position_embeddings,
                                   "--ctx-size exceeds max_position_embeddings");
        c.max_position_embeddings = m.max_position_embeddings = context;
    }
    return model;
}

void load(Model& model, const weight_loader::Loader& weights, LayerPlacementConfig& placement,
          Collectives& collectives, int primary, const std::unordered_set<int>& offload,
          int64_t& host_bytes, int gpu_layers) {
    const auto& c = model.config;
    placement.prepare(c.num_hidden_layers);
    const int cpu_layers = gpu_layers < 0 ? 0 :
        static_cast<int>(std::max<int64_t>(c.num_hidden_layers - gpu_layers, 0));
    if (cpu_layers > 0) {
        if (weights.placement.tensor.enabled() || weights.placement.experts.enabled() || placement.enabled())
            throw std::runtime_error("--n-gpu-layers cannot be combined with tensor/expert/layer parallelism");
        throw std::runtime_error("--n-gpu-layers currently supports dense Qwen-style blocks only");
    }
    MfqCudaGuard guard(placement.primary_device());
    const bool separate = !c.tie_word_embeddings && weights.has("model.output.weight");
    model.embed = weights.embedding("model.token_embedding.weight", separate);
    model.output_norm = weights.tensor("model.output_norm.weight");
    model.output_head = load_output_head(weights);
    model.lm_head = separate ? weights.weight("model.output.weight", TensorParallelAxis::Output)
        : weights.placement.tensor.enabled()
            ? weights.weight("model.token_embedding.weight", TensorParallelAxis::Output)
            : model.embed;
    model.layers.reserve(static_cast<size_t>(c.num_hidden_layers));
    for (int index = 0; index < c.num_hidden_layers; ++index) {
        const int device = placement.device_for_layer(index);
        const auto& type = model.metadata.layer_types.at(static_cast<size_t>(index));
        std::cerr << "loading layer " << index << ' ' << type << " CUDA" << std::endl;
        model.layers.push_back(load_layer(weights.for_layer(device, false), c, index, type,
                                          collectives, primary, offload, host_bytes));
    }
}

namespace {
DenseTensor normalize_ids(Collectives& collectives, int primary, DenseTensor ids) {
    tyloi::models::require_model(ids.dim() == 1 || ids.dim() == 2,
                               "token IDs must have shape [tokens] or [batch,tokens]");
    if (ids.dim() == 1) ids = ids.unsqueeze(0);
    tyloi::models::require_model(ids.size(0) > 0 && ids.size(1) > 0,
                               "token IDs must not be empty");
    return collectives.copy(ids.to(::tyloi::kInt64), primary).contiguous();
}
} // namespace

DenseTensor embed(const Model& model, Executor& executor, Collectives& collectives, int primary,
             DenseTensor ids, const Executor::Mode& mode) {
    const Task task{task::Embed{0, 0, *model.embed}, -1, {}, "model.embed"};
    Inputs inputs;
    inputs.values = {normalize_ids(collectives, primary, std::move(ids))};
    return executor.run(std::span<const Task>{&task, 1}, std::move(inputs), mode);
}

DenseTensor normalize(const Model& model, Executor& executor, DenseTensor hidden, int64_t batch, int64_t tokens) {
    return executor.profiler().measure("model.output_norm", [&]() {
        return qwen_rms_norm(hidden.reshape({batch * tokens, model.metadata.hidden_size})
                                 .to(::tyloi::kFloat32),
                             model.output_norm, model.metadata.rms_norm_eps,
                             model.metadata.norm_weight_offset)
            .reshape({batch, tokens, model.metadata.hidden_size});
    });
}

DenseTensor softcap(const Model& model, Executor&, DenseTensor value) {
    const double cap = model.metadata.final_logit_softcapping;
    return cap > 0.0 ? ::tyloi::cuda::tanh(value / cap) * cap : value;
}

DenseTensor logits(const Model& model, Executor& executor, DenseTensor hidden, bool capped,
              const Executor::Mode& mode) {
    const Task task{task::Project{0, 0, *model.lm_head}, -1, {}, "model.lm_head"};
    Inputs inputs;
    inputs.values = {std::move(hidden)};
    auto value = executor.run(std::span<const Task>{&task, 1}, std::move(inputs), mode);
    return capped ? softcap(model, executor, std::move(value)) : value;
}

DenseTensor forward(const Model& model, State& state, Executor& executor, Collectives& collectives,
               const Executor::Mode& mode, Input input) {
    MfqCudaGuard guard(mode.primary);
    auto ids = normalize_ids(collectives, mode.primary, DenseTensor(std::move(input.ids)));
    auto embeddings = input.embeddings.has_value() ? DenseTensor(std::move(*input.embeddings))
        : embed(model, executor, collectives, mode.primary, ids, mode);
    const auto B = ids.size(0), T = ids.size(1);
    tyloi::models::require_model(embeddings.dim() == 3 && embeddings.size(0) == B &&
                                   embeddings.size(1) == T && embeddings.size(2) == model.metadata.hidden_size,
                               "inputs_embeds shape must match [batch,tokens,hidden_size]");
    tyloi::models::require_model(input.confirmed == 0, "unsupported speculative backbone geometry");
    if (state.cache_pos == 0) state.reset(B);
    const auto options = ::tyloi::TensorOptions().device(::tyloi::Device(::tyloi::kCUDA, mode.primary)).dtype(::tyloi::kInt64);
    auto physical = input.cache_positions.has_value()
        ? collectives.copy(DenseTensor(*input.cache_positions).to(::tyloi::kInt64), mode.primary).contiguous()
        : ::tyloi::cuda::arange(state.cache_pos, state.cache_pos + T, options);
    auto semantic = input.positions.has_value()
        ? collectives.copy(DenseTensor(*input.positions).to(::tyloi::kInt64), mode.primary).contiguous()
        : (state.decode_position_delta == 0 ? physical : physical + state.decode_position_delta);
    tyloi::models::require_model((physical.dim() == 1 && physical.numel() == T) ||
                                   (physical.dim() == 2 && physical.size(0) == B && physical.size(1) == T),
                               "cache_positions must have shape [tokens] or [batch,tokens]");
    tyloi::models::require_model((semantic.dim() == 1 && semantic.size(0) == T) ||
                                   (semantic.dim() == 2 && semantic.size(0) == B && semantic.size(1) == T),
                               "position_ids must have shape [tokens] or [batch,tokens]");
    if (input.attention_mask.has_value()) {
        const auto& mask = *input.attention_mask;
        tyloi::models::require_model(mask.dim() == 2 && mask.size(0) == B && mask.size(1) >= state.cache_pos + T,
                                   "attention_mask must cover [batch,cached_plus_current_tokens]");
    }
    if (!input.lengths.has_value() && state.cache_pos > 0 && T == 1)
        input.lengths = ::tyloi::cuda::full({B}, state.cache_pos + 1, options);
    AttentionStep step{semantic, input.lengths ? MfqOptional<DenseTensor>(DenseTensor(*input.lengths)) : mfq_nullopt,
                       input.cache_positions.has_value() ? MfqOptional<DenseTensor>(physical) : mfq_nullopt,
                       {}, state.cache_pos, input.kv_length, input.attention_parts};
    auto hidden = collectives.copy(std::move(embeddings), mode.primary).contiguous()
                      .to(::tyloi::kFloat16).unsqueeze(2)
                      .expand({B, T, model.metadata.hc_mult, model.metadata.hidden_size}).contiguous();
    const auto trace = [&](const DenseTensor& value) {
        if (input.trace) input.trace->push_back(collectives.copy(value, mode.primary).to(::tyloi::kFloat32).clone());
    };
    trace(hidden);
    for (std::size_t index = 0; index < model.layers.size(); ++index) {
        const auto& layer = model.layers[index];
        MfqCudaGuard layer_guard(layer.device);
        auto local = step;
        local.positions = collectives.copy(DenseTensor(step.positions), layer.device);
        if (step.sequence_lengths.has_value()) local.sequence_lengths = collectives.copy(DenseTensor(*step.sequence_lengths), layer.device);
        if (step.cache_positions.has_value()) local.cache_positions = collectives.copy(DenseTensor(*step.cache_positions), layer.device);
        hidden = forward_layer(layer, state.layers[index], executor,
                               collectives.copy(std::move(hidden), layer.device),
                               collectives.copy(ids, layer.device), local, mode);
        trace(hidden);
    }
    if (!input.positions.has_value() || input.advance) state.cache_pos += T;
    hidden = collectives.copy(std::move(hidden), mode.primary);
    auto collapsed = finalize_hidden(hidden, model.output_head, model.config, executor.profiler(), B, T);
    auto finalized = normalize(model, executor, std::move(collapsed), B, T);
    if (input.raw_hidden) *input.raw_hidden = std::move(hidden);
    return finalized;
}

} // namespace tyloi::cuda::deepseek_v4

bool g_dsv4_fused_hc = true;
bool g_dsv4_compare_hc_ops = false;

std::vector<::tyloi::DenseTensor>
dsv4_hc_split_sinkhorn(::tyloi::DenseTensor mixes, ::tyloi::DenseTensor scale,
                       ::tyloi::DenseTensor base, int64_t hc_mult, int64_t iterations,
                       double eps) {
    const int64_t prefix = 2 * hc_mult;
    auto pre = ::tyloi::cuda::sigmoid(mixes.slice(-1, 0, hc_mult) * scale.index({0}) +
                                           base.slice(0, 0, hc_mult)) +
               eps;
    auto post =
        2.0 * ::tyloi::cuda::sigmoid(mixes.slice(-1, hc_mult, prefix) * scale.index({1}) +
                                          base.slice(0, hc_mult, prefix));
    auto comb = (mixes.slice(-1, prefix, mixes.size(-1)) * scale.index({2}) +
                 base.slice(0, prefix, base.size(0)))
                    .reshape({mixes.size(0), mixes.size(1), hc_mult, hc_mult});
    comb = ::tyloi::cuda::softmax(comb, -1) + eps;
    comb = comb / (comb.sum(-2, true) + eps);
    for (int64_t i = 1; i < iterations; ++i) {
        comb = comb / (comb.sum(-1, true) + eps);
        comb = comb / (comb.sum(-2, true) + eps);
    }
    return {pre, post, comb};
}


static void dsv4_report_hc_difference(int layer, const char *stage, const char *tensor_name,
                                      const ::tyloi::DenseTensor &reference,
                                      const ::tyloi::DenseTensor &candidate) {
    auto reference_f32 = reference.to(::tyloi::kFloat32);
    auto candidate_f32 = candidate.to(::tyloi::kFloat32);
    auto difference = candidate_f32 - reference_f32;
    const double reference_norm = std::max(reference_f32.norm().item<double>(), 1.0e-30);
    std::cout << std::scientific << std::setprecision(9) << "dsv4_hc_op_ab layer=" << layer
              << " stage=" << stage << " tensor=" << tensor_name
              << " differing=" << candidate.ne(reference).sum().item<int64_t>()
              << " rel_l2=" << difference.norm().item<double>() / reference_norm
              << " mean_abs=" << difference.abs().mean().item<double>()
              << " max_abs=" << difference.abs().max().item<double>() << "\n";
}


namespace tyloi::models::deepseek_v4 {
template <class DenseTensor> struct Projections {
    DenseTensor rank, queries, values;
    std::vector<DenseTensor> compressor, indexer;
};

template <class Query, class Values, class Compressor, class Indexer, class Prefill, class Decode,
          class Concat>
auto attention(int64_t tokens, int64_t position, bool lengths, int64_t ratio, Query query,
               Values values, Compressor compressor, Indexer indexer, Prefill prefill,
               Decode decode, Concat concat) {
    auto q = query();
    using DenseTensor = decltype(values());
    Projections<DenseTensor> projected{q[0], q[1], values(), {}, {}};
    if (ratio > 0)
        projected.compressor = compressor();
    if (ratio == 4)
        projected.indexer = indexer();
    if (tokens > 1 && position == 0 && !lengths)
        return prefill(projected);
    std::vector<DenseTensor> output;
    output.reserve(tokens);
    for (int64_t token = 0; token < tokens; ++token)
        output.push_back(decode(projected, token, position + token + 1));
    return concat(output);
}

template <class Index, class All, class Empty>
auto select_compressed(int64_t ratio, int64_t visible, Index index, All all, Empty empty) {
    if (ratio == 4 && visible > 512)
        return index();
    if (visible > 0)
        return all();
    return empty();
}

template <class Write, class Compress, class Index, class Select, class Plan, class Cache,
          class Attend, class Inverse, class Output>
auto compressed_step(int64_t ratio, bool prefill, Write write, Compress compress, Index index,
                     Select select, Plan plan, Cache cache, Attend attend, Inverse inverse,
                     Output output) {
    write();
    const auto visible = ratio > 0 ? compress() : 0;
    if (ratio == 4) {
        const auto indexed = index();
        require_model(!prefill || indexed == visible,
                      "DeepSeek V4 compressor/indexer prefill length mismatch");
    }
    auto selected = select(visible);
    auto indices = plan(std::move(selected), visible);
    auto values = cache(visible);
    return output(inverse(attend(std::move(values), indices)));
}

} // namespace tyloi::models::deepseek_v4

namespace tyloi::cuda::deepseek_v4 {
using tyloi::models::deepseek_v4::PoolSnapshot;

static PoolSnapshot capture_pool(const PoolState &source, int64_t cache_pos, size_t &bytes) {
    PoolSnapshot state;
    state.ratio = source.ratio;
    state.head_dim = source.head_dim;
    state.cache_quant_mode = source.cache_quant_mode;
    state.capacity = source.capacity;
    state.overlap = source.overlap;
    if (source.ratio <= 0)
        return state;
    if (source.capacity <= 0 || !source.state_kv.defined() || !source.state_gate.defined() ||
        !source.pool.defined() || source.pool.dim() != 3 || source.pool.size(0) != 1 ||
        source.pool.size(1) != source.capacity ||
        (source.overlap && (!source.previous_kv.defined() || !source.previous_gate.defined()))) {
        throw std::runtime_error("DeepSeek V4 session compressor state is unavailable");
    }
    const int64_t visible = std::min<int64_t>(cache_pos / source.ratio, source.capacity);
    state.state_kv = std::make_shared<DenseTensor>(source.state_kv.clone());
    state.state_gate = std::make_shared<DenseTensor>(source.state_gate.clone());
    if (source.overlap) {
        state.previous_kv = std::make_shared<DenseTensor>(source.previous_kv.clone());
        state.previous_gate = std::make_shared<DenseTensor>(source.previous_gate.clone());
    }
    state.pool = std::make_shared<DenseTensor>(source.pool.narrow(1, 0, visible).clone());
    bytes += session_tensor_bytes(*state.state_kv);
    bytes += session_tensor_bytes(*state.state_gate);
    if (state.previous_kv) bytes += session_tensor_bytes(*state.previous_kv);
    if (state.previous_gate) bytes += session_tensor_bytes(*state.previous_gate);
    bytes += session_tensor_bytes(*state.pool);
    return state;
}

static void restore_pool(PoolState &target, const PoolSnapshot &state) {
    if (target.ratio != state.ratio || target.head_dim != state.head_dim ||
        target.cache_quant_mode != state.cache_quant_mode || target.overlap != state.overlap) {
        throw CudaSessionStateError("DeepSeek V4 session compressor configuration changed");
    }
    if (state.ratio <= 0)
        return;
    if (state.capacity <= 0 || !state.state_kv || !state.state_kv->defined() ||
        !state.state_gate || !state.state_gate->defined() ||
        !state.pool || !state.pool->defined() || state.pool->dim() != 3 || state.pool->size(0) != 1 ||
        state.pool->size(1) > state.capacity ||
        (state.overlap && (!state.previous_kv || !state.previous_kv->defined() ||
                           !state.previous_gate || !state.previous_gate->defined()))) {
        throw CudaSessionStateError("DeepSeek V4 saved compressor state is invalid");
    }
    target.capacity = state.capacity;
    restore_session_tensor(target.state_kv, state.state_kv.get());
    restore_session_tensor(target.state_gate, state.state_gate.get());
    if (state.overlap) {
        restore_session_tensor(target.previous_kv, state.previous_kv.get());
        restore_session_tensor(target.previous_gate, state.previous_gate.get());
    } else {
        target.previous_kv = ::tyloi::DenseTensor();
        target.previous_gate = ::tyloi::DenseTensor();
    }
    restore_session_prefix_tensor(target.pool, *state.pool, 1, state.capacity);
}

void PoolState::reset(int64_t batch, int64_t max_positions) {
    if (ratio <= 0 || head_dim <= 0)
        return;
    capacity = std::max<int64_t>(1, (max_positions + ratio - 1) / ratio);
    const int64_t output_dim = overlap ? 2 * head_dim : head_dim;
    auto fp32 = ::tyloi::TensorOptions()
                    .device(::tyloi::kCUDA)
                    .dtype(::tyloi::kFloat32);
    auto half = fp32.dtype(::tyloi::kFloat16);
    if (state_kv.defined() && state_kv.size(0) == batch && pool.size(1) == capacity) {
        state_kv.zero_();
        state_gate.fill_(-std::numeric_limits<float>::infinity());
        if (overlap) {
            previous_kv.zero_();
            previous_gate.fill_(-std::numeric_limits<float>::infinity());
        }
        pool.zero_();
        return;
    }
    state_kv = ::tyloi::cuda::zeros({batch, ratio, output_dim}, fp32);
    state_gate = ::tyloi::cuda::full({batch, ratio, output_dim},
                                          -std::numeric_limits<float>::infinity(), fp32);
    if (overlap) {
        previous_kv = ::tyloi::cuda::zeros({batch, ratio, head_dim}, fp32);
        previous_gate = ::tyloi::cuda::full({batch, ratio, head_dim},
                                                 -std::numeric_limits<float>::infinity(), fp32);
    } else {
        previous_kv = ::tyloi::cuda::empty({0}, fp32);
        previous_gate = ::tyloi::cuda::empty({0}, fp32);
    }
    pool = ::tyloi::cuda::zeros({batch, capacity, head_dim}, half);
}

std::vector<::tyloi::DenseTensor> Compressor::project(::tyloi::DenseTensor x, int64_t batch,
                                                int64_t tokens) const {
    auto parts = projection.forward(x.to(::tyloi::kFloat32));
    const int64_t output_dim = overlap ? 2 * head_dim : head_dim;
    return {
        parts.at(0)
            .reshape({batch, tokens, output_dim})
            .to(::tyloi::kFloat32)
            .contiguous(),
        parts.at(1)
            .reshape({batch, tokens, output_dim})
            .to(::tyloi::kFloat32)
            .contiguous(),
    };
}

void Compressor::update(PoolState& state, ::tyloi::DenseTensor kv, ::tyloi::DenseTensor gate,
            ::tyloi::DenseTensor length, const tyloi::cuda::TailRope &rope) const {
    dsv4_decode_pool_update_cuda(kv, gate, ape, norm, state.state_kv, state.state_gate, state.previous_kv,
                                 state.previous_gate, state.pool, length, rope.cos(), rope.sin(), ratio,
                                 overlap, cache_quant_mode, 1e-6);
}

int64_t Compressor::prefill(PoolState& state, const ::tyloi::DenseTensor &kv, const ::tyloi::DenseTensor &gate,
                const tyloi::cuda::TailRope &rope) const {
    if (ratio <= 0)
        return 0;
    if (kv.dim() != 3 || gate.sizes() != kv.sizes()) {
        throw std::runtime_error(
            "DeepSeek V4 compressor prefill expects matching [B,T,D] tensors");
    }
    const int64_t batch = kv.size(0);
    const int64_t tokens = kv.size(1);
    const int64_t output_dim = overlap ? 2 * head_dim : head_dim;
    if (kv.size(2) != output_dim) {
        throw std::runtime_error("DeepSeek V4 compressor prefill projection width mismatch");
    }
    const int64_t windows = tokens / ratio;
    const int64_t cutoff = windows * ratio;
    if (windows > 0) {
        auto grouped_kv =
            kv.narrow(1, 0, cutoff).reshape({batch, windows, ratio, output_dim}).contiguous();
        auto grouped_gate =
            gate.narrow(1, 0, cutoff).reshape({batch, windows, ratio, output_dim}).contiguous();
        auto positions =
            ::tyloi::cuda::arange(windows, ::tyloi::TensorOptions()
                                                    .device(kv.device())
                                                    .dtype(::tyloi::kInt64))
                .reshape({1, windows})
                .expand({batch, windows})
                .contiguous();
        auto empty = ::tyloi::cuda::empty({0}, kv.options());
        auto compressed =
            dsv4_compress_cuda(grouped_kv, grouped_gate, ape, norm, empty, empty, positions,
                               rope.cos(), rope.sin(), ratio, overlap, cache_quant_mode, 1e-6);
        state.pool.narrow(1, 0, windows).copy_(compressed);

        state.state_kv.copy_(kv.narrow(1, cutoff - ratio, ratio).contiguous());
        state.state_gate.copy_(gate.narrow(1, cutoff - ratio, ratio).contiguous());
        if (overlap) {
            state.previous_kv.copy_(state.state_kv.narrow(2, 0, head_dim));
            state.previous_gate.copy_(state.state_gate.narrow(2, 0, head_dim));
        }
    }
    const int64_t remainder = tokens - cutoff;
    if (remainder > 0) {
        state.state_kv.narrow(1, 0, remainder).copy_(kv.narrow(1, cutoff, remainder));
        state.state_gate.narrow(1, 0, remainder).copy_(gate.narrow(1, cutoff, remainder));
    }
    return windows;
}

State::State(const Model& model) : layers(model.layers.size()) {
    layout_.reserve(model.layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& layer = model.layers[i];
        layout_.push_back({layer.device, layer.head_dim, layer.max_positions});
        const auto init = [](PoolState& state, const Compressor& weight) {
            state.ratio = weight.ratio;
            state.head_dim = weight.head_dim;
            state.cache_quant_mode = weight.cache_quant_mode;
            state.overlap = weight.overlap;
        };
        init(layers[i].compressor, layer.compressor);
        init(layers[i].indexer_compressor, layer.indexer_compressor);
    }
}

void State::reset(int64_t next_batch) {
    cache_pos = decode_position_delta = 0;
    batch = next_batch;
    for (size_t i = 0; i < layers.size(); ++i) {
        MfqCudaGuard guard(layout_[i].device);
        auto& layer = layers[i];
        auto half = ::tyloi::TensorOptions().device(::tyloi::kCUDA)
                        .dtype(::tyloi::kFloat16);
        if (!layer.local_cache.defined() || layer.local_cache.size(0) != next_batch)
            layer.local_cache = ::tyloi::cuda::zeros({next_batch, 128, layout_[i].head_dim}, half);
        else layer.local_cache.zero_();
        layer.compressor.reset(next_batch, layout_[i].max_positions);
        layer.indexer_compressor.reset(next_batch, layout_[i].max_positions);
    }
}

std::vector<DenseTensor> State::storage() const {
    std::vector<DenseTensor> result;
    const auto append = [&](const DenseTensor& tensor) { if (tensor.defined()) result.push_back(tensor); };
    const auto pool = [&](const PoolState& state) {
        append(state.state_kv); append(state.state_gate); append(state.previous_kv);
        append(state.previous_gate); append(state.pool);
    };
    for (const auto& layer : layers) {
        append(layer.local_cache); pool(layer.compressor); pool(layer.indexer_compressor);
    }
    return result;
}

TextSessionStateKind State::text_session_state_kind() const {
    return layers.empty() ? TextSessionStateKind::Unsupported : TextSessionStateKind::DeepseekV4;
}
bool State::supports_text_session_state() const {
    return text_session_state_kind() != TextSessionStateKind::Unsupported;
}

TextSessionState State::capture_text_session_state(const std::vector<int64_t>& tokens) const {
    if (!supports_text_session_state() || cache_pos <= 0 || static_cast<size_t>(cache_pos) != tokens.size())
        throw std::runtime_error("DeepSeek-V4 text session state is unavailable");
    TextSessionState state;
    state.tokens = tokens;
    state.cache_pos = cache_pos;
    state.decode_position_delta = decode_position_delta;
    state.payload = std::vector<tyloi::models::deepseek_v4::BlockSnapshot>{};
    auto& saved_layers = std::get<std::vector<tyloi::models::deepseek_v4::BlockSnapshot>>(state.payload);
    saved_layers.reserve(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        MfqCudaGuard guard(layout_[i].device);
        const auto& layer = layers[i];
        if (!layer.local_cache.defined())
            throw std::runtime_error("DeepSeek V4 local session cache is unavailable");
        tyloi::models::deepseek_v4::BlockSnapshot saved;
        saved.local_cache = std::make_shared<DenseTensor>(layer.local_cache.clone());
        state.bytes += session_tensor_bytes(*saved.local_cache);
        saved.compressor = capture_pool(layer.compressor, cache_pos, state.bytes);
        saved.indexer_compressor = capture_pool(layer.indexer_compressor, cache_pos, state.bytes);
        saved_layers.push_back(std::move(saved));
    }
    return state;
}

void State::restore_text_session_state(const TextSessionState& state) {
    const auto* saved_layers = std::get_if<std::vector<tyloi::models::deepseek_v4::BlockSnapshot>>(&state.payload);
    if (!supports_text_session_state() || state.kind() != TextSessionStateKind::DeepseekV4 ||
        state.cache_pos <= 0 || static_cast<size_t>(state.cache_pos) != state.tokens.size() ||
        saved_layers == nullptr || saved_layers->size() != layers.size())
        throw CudaSessionStateError("DeepSeek V4 text session state is incompatible");
    for (size_t i = 0; i < layers.size(); ++i) {
        MfqCudaGuard guard(layout_[i].device);
        auto& layer = layers[i];
        const auto& saved = (*saved_layers)[i];
        if (!saved.local_cache || !saved.local_cache->defined() || saved.local_cache->dim() != 3 ||
            saved.local_cache->size(0) != 1 || saved.local_cache->size(1) != 128)
            throw CudaSessionStateError("DeepSeek V4 saved local cache is invalid");
        restore_session_tensor(layer.local_cache, saved.local_cache.get());
        restore_pool(layer.compressor, saved.compressor);
        restore_pool(layer.indexer_compressor, saved.indexer_compressor);
    }
    cache_pos = state.cache_pos;
    decode_position_delta = state.decode_position_delta;
    batch = 1;
}

static std::vector<::tyloi::DenseTensor> hc_pre(const Layer& w, ::tyloi::DenseTensor x,
                                               ::tyloi::DenseTensor function,
                                               ::tyloi::DenseTensor scale,
                                               ::tyloi::DenseTensor base,
                                               const char *stage) {
    const auto& eps = w.eps;
    const auto& hc_iterations = w.hc_iterations;
    const auto& hc_eps = w.hc_eps;
    const auto& hc_mult = w.hc_mult;
    const auto& layer = w.layer;

    auto flat = x.flatten(2).to(::tyloi::kFloat32);
    auto inverse_rms = ::tyloi::cuda::rsqrt(flat.square().mean(-1, true) + eps);
    auto mixes = ::tyloi::cuda::matmul(flat, function.transpose(0, 1)) * inverse_rms;
    std::vector<::tyloi::DenseTensor> candidate;
    if (g_dsv4_fused_hc || g_dsv4_compare_hc_ops) {
        candidate = dsv4_hc_pre_cuda(x, mixes.contiguous(), scale, base, hc_iterations, hc_eps);
    }
    std::vector<::tyloi::DenseTensor> reference;
    if (!g_dsv4_fused_hc || g_dsv4_compare_hc_ops) {
        auto split = dsv4_hc_split_sinkhorn(mixes, scale, base, hc_mult, hc_iterations, hc_eps);
        auto reduced = (split.at(0).unsqueeze(-1) * flat.reshape(x.sizes()))
                           .sum(2)
                           .to(::tyloi::kFloat16)
                           .contiguous();
        reference = {reduced, split.at(1), split.at(2)};
    }
    if (g_dsv4_compare_hc_ops && layer == 0) {
        dsv4_report_hc_difference(layer, stage, "reduced", reference.at(0), candidate.at(0));
        dsv4_report_hc_difference(layer, stage, "post", reference.at(1), candidate.at(1));
        dsv4_report_hc_difference(layer, stage, "combination", reference.at(2),
                                  candidate.at(2));
    }
    return g_dsv4_fused_hc ? candidate : reference;
}

static ::tyloi::DenseTensor hc_post(const Layer& w, ::tyloi::DenseTensor x,
                                   ::tyloi::DenseTensor residual,
                                   ::tyloi::DenseTensor post,
                                   ::tyloi::DenseTensor combination,
                                   const char *stage) {
    const auto& layer = w.layer;

    ::tyloi::DenseTensor candidate;
    if (g_dsv4_fused_hc || g_dsv4_compare_hc_ops) {
        candidate = dsv4_hc_post_cuda(x.contiguous(), residual.contiguous(), post.contiguous(),
                                      combination.contiguous());
    }
    ::tyloi::DenseTensor reference;
    if (!g_dsv4_fused_hc || g_dsv4_compare_hc_ops) {
        reference = (post.unsqueeze(-1) * x.unsqueeze(-2) +
                     (combination.unsqueeze(-1) *
                      residual.to(::tyloi::kFloat32).unsqueeze(-2))
                         .sum(2))
                        .to(::tyloi::kFloat16)
                        .contiguous();
    }
    if (g_dsv4_compare_hc_ops && layer == 0) {
        dsv4_report_hc_difference(layer, stage, "expanded", reference, candidate);
    }
    return g_dsv4_fused_hc ? candidate : reference;
}

static ::tyloi::DenseTensor
attention_forward(const Layer& w, LayerState& state, Executor& executor,
                  ::tyloi::DenseTensor x, const AttentionStep& step,
                  const Executor::Mode& mode) {
    auto &profiler = executor.profiler();
    auto project = executor.projector(mode.ffn);
    auto& scratch = executor.mla_scratch(w.device);
    scratch.ensure_compressed();
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    const ::tyloi::DenseTensor positions(step.positions);
    const MfqOptional<::tyloi::DenseTensor> seq_len = step.sequence_lengths
        ? MfqOptional<::tyloi::DenseTensor>(::tyloi::DenseTensor(*step.sequence_lengths)) : mfq_nullopt;
#else
    const auto& positions = step.positions;
    const auto& seq_len = step.sequence_lengths;
#endif
    const auto cache_pos = step.cache_position;
    auto& local_cache = state.local_cache;
    const auto& hidden_size = w.hidden_size;
    const auto& compress_ratio = w.compress_ratio;
    const auto& heads = w.heads;
    const auto& head_dim = w.head_dim;
    const auto& eps = w.eps;
    const auto& q_a = *w.q_a;
    const auto& q_a_norm = w.q_a_norm;
    const auto& q_b = *w.q_b;
    const auto& attention_rope = w.attention_rope;
    const auto& kv = *w.kv;
    const auto& kv_norm = w.kv_norm;
    const auto& compressor = w.compressor;
    const auto& indexer_compressor = w.indexer_compressor;
    const auto& indexer_q = w.indexer_q;
    const auto& indexer_weight = w.indexer_weight;
    const auto& sinks = w.sinks;
    const auto& output_a = *w.output_a;
    const auto& output_b = *w.output_b;
    const auto& groups = w.groups;

    const int64_t batch = x.size(0);
    const int64_t tokens = x.size(1);
    auto flat =
        x.reshape({batch * tokens, hidden_size}).to(::tyloi::kFloat16).contiguous();

    using DenseTensor = ::tyloi::DenseTensor;
    return tyloi::models::deepseek_v4::attention(
        tokens, cache_pos, seq_len.has_value(), compress_ratio,
        [&] {
            auto qr =
                profiler.measure("dsv4.q_a", [&]() { return project.run(q_a, flat); });
            qr = profiler.measure("dsv4.q_a_norm", [&]() {
                return rms_norm_cuda(
                           qr.reshape({-1, qr.size(-1)}).to(::tyloi::kFloat32),
                           q_a_norm, eps)
                    .to(::tyloi::kFloat16)
                    .contiguous();
            });
            auto queries = profiler.measure("dsv4.q_b", [&]() {
                return project.run(q_b, qr)
                    .reshape({batch, tokens, heads, head_dim})
                    .transpose(1, 2)
                    .contiguous()
                    .to(::tyloi::kFloat32);
            });
            queries = profiler.measure("dsv4.q_norm_rope", [&]() {
                auto normalized =
                    queries * ::tyloi::cuda::rsqrt(queries.square().mean(-1, true) + eps);
                return attention_rope.forward(normalized, positions, false);
            });

            return std::array<DenseTensor, 2>{qr, queries};
        },
        [&] {
            auto values = profiler.measure("dsv4.kv", [&]() {
                return project.run(kv, flat).reshape({batch, tokens, head_dim});
            });
            values = profiler.measure("dsv4.kv_norm_rope", [&]() {
                auto normalized =
                    rms_norm_cuda(
                        values.reshape({-1, head_dim}).to(::tyloi::kFloat32),
                        kv_norm, eps)
                        .reshape({batch, tokens, head_dim})
                        .to(::tyloi::kFloat16);
                auto values_heads = normalized.unsqueeze(1).contiguous();
                values_heads =
                    attention_rope.forward(values_heads, positions, false);
                return values_heads.squeeze(1).contiguous();
            });

            return values;
        },
        [&] {
            return profiler.measure("dsv4.compressor_proj",
                                    [&] { return compressor.project(flat, batch, tokens); });
        },
        [&] {
            return profiler.measure("dsv4.indexer_compressor_proj", [&] {
                return indexer_compressor.project(flat, batch, tokens);
            });
        },
        [&](const auto &projected) {
            auto qr = projected.rank, queries = projected.queries, values = projected.values;
            const auto &compressor_parts = projected.compressor;
            const auto &indexer_parts = projected.indexer;
            return tyloi::models::deepseek_v4::compressed_step(
                compress_ratio, true,
                [&] {
                    const int64_t local_tokens = std::min<int64_t>(tokens, 128);
                    auto local_positions =
                        positions.narrow(0, tokens - local_tokens, local_tokens)
                            .remainder(128)
                            .to(::tyloi::kInt64)
                            .contiguous();
                    profiler.measure("dsv4.local_cache_prefill", [&]() {
                        local_cache.index_copy_(
                            1, local_positions,
                            values.narrow(1, tokens - local_tokens, local_tokens));
                        return 0;
                    });
                },
                [&] {
                    return profiler.measure("dsv4.compressor_prefill", [&] {
                        return compressor.prefill(state.compressor, compressor_parts.at(0),
                                                  compressor_parts.at(1), attention_rope);
                    });
                },
                [&] {
                    return profiler.measure("dsv4.indexer_compressor_prefill", [&] {
                        return indexer_compressor.prefill(state.indexer_compressor, indexer_parts.at(0),
                                                          indexer_parts.at(1), attention_rope);
                    });
                },
                [&](int64_t visible) {
                    auto int_options = ::tyloi::TensorOptions()
                                           .device(x.device())
                                           .dtype(::tyloi::kInt32);
                    return tyloi::models::deepseek_v4::select_compressed(
                        compress_ratio, visible,
                        [&] {
                            auto index_query =
                                profiler.measure("dsv4.indexer_q_prefill", [&]() {
                                    return project.run(*indexer_q, qr)
                                        .reshape({batch, tokens, heads, 128})
                                        .transpose(1, 2)
                                        .contiguous();
                                });
                            index_query = attention_rope.forward(index_query, positions, false)
                                              .transpose(1, 2)
                                              .contiguous();
                            index_query = nepq_hadamard_input_cuda(
                                              index_query.reshape({batch * tokens * heads, 128})
                                                  .to(::tyloi::kFloat16)
                                                  .contiguous(),
                                              scratch.hadamard_signs, 128)
                                              .reshape({batch, tokens, heads, 128});
                            index_query = dsv4_fp4_sim_cuda(index_query.contiguous());
                            auto weights =
                                profiler.measure("dsv4.indexer_weight_prefill", [&]() {
                                    return ::tyloi::cuda::matmul(
                                               x.reshape({batch * tokens, hidden_size})
                                                   .to(::tyloi::kFloat32),
                                               indexer_weight.transpose(0, 1))
                                        .reshape({batch, tokens, heads})
                                        .to(::tyloi::kFloat16)
                                        .contiguous();
                                });
                            auto scores =
                                profiler.measure("dsv4.indexer_scores_prefill", [&]() {
                                    return dsv4_indexer_scores_cuda(
                                        index_query,
                                        state.indexer_compressor.pool.narrow(1, 0, visible)
                                            .contiguous(),
                                        weights, 0, 4);
                                });
                            return profiler.measure("dsv4.indexer_topk_prefill", [&]() {
                                return dsv4_topk512_cuda(scores);
                            });
                        },
                        [&] {
                            return ::tyloi::cuda::arange(visible, int_options)
                                .reshape({1, 1, visible})
                                .expand({batch, tokens, visible})
                                .contiguous();
                        },
                        [&] {
                            return ::tyloi::cuda::zeros({batch, tokens, 1}, int_options);
                        });
                },
                [&](DenseTensor selected, int64_t visible) {
                    const int64_t plan_ratio = compress_ratio > 0 ? compress_ratio : 1;
                    return profiler.measure("dsv4.attention_plan_prefill", [&]() {
                        return dsv4_build_prefill_plan_cuda(selected, 0, 0, visible, plan_ratio,
                                                            128);
                    });
                },
                [&](int64_t visible) {
                    return profiler.measure("dsv4.attention_cache_prefill", [&]() {
                        return visible > 0
                                   ? ::tyloi::cuda::cat(
                                         {values, state.compressor.pool.narrow(1, 0, visible)}, 1)
                                         .contiguous()
                                   : values.contiguous();
                    });
                },
                [&](DenseTensor cache, const auto &plan) {
                    return profiler.measure("dsv4.sparse_attention_prefill", [&]() {
                        return attention_dsv4_sparse_cuda(
                            queries, cache, plan.at(0), plan.at(1), sinks,
                            scratch.attention_meta,
                            1.0 / std::sqrt(static_cast<double>(head_dim)));
                    });
                },
                [&](DenseTensor attention) {
                    return profiler.measure("dsv4.attention_inverse_rope_prefill", [&]() {
                        auto transposed = attention.transpose(1, 2).contiguous();
                        transposed =
                            attention_rope.forward(transposed, positions, true);
                        return transposed.transpose(1, 2).contiguous();
                    });
                },
                [&](DenseTensor attention) { return project.low_rank(attention, output_a, output_b, groups, mode.groupwise_output, true); });
        },
        [&](const auto &projected, int64_t token, int64_t absolute_length) {
            auto qr = projected.rank, queries = projected.queries, values = projected.values;
            const auto &compressor_parts = projected.compressor;
            const auto &indexer_parts = projected.indexer;
            auto position = positions.narrow(0, token, 1);
            auto slot = ::tyloi::cuda::remainder(position, 128)
                            .to(::tyloi::kInt64)
                            .contiguous();
            ::tyloi::DenseTensor length;
            if (tokens == 1 && seq_len.has_value()) {
                length = seq_len.value().contiguous();
            } else {
                length = ::tyloi::cuda::full({batch}, absolute_length,
                                                  ::tyloi::TensorOptions()
                                                      .device(::tyloi::kCUDA)
                                                      .dtype(::tyloi::kInt64));
            }
            return tyloi::models::deepseek_v4::compressed_step(
                compress_ratio, false,
                [&] {
                    profiler.measure("dsv4.local_cache_write", [&]() {
                        glm_dsa_cache_write_cuda(local_cache,
                                                 values.narrow(1, token, 1).contiguous(), slot);
                        return 0;
                    });
                },
                [&] {
                    profiler.measure("dsv4.compressor_update", [&]() {
                        compressor.update(
                            state.compressor, compressor_parts.at(0).narrow(1, token, 1).contiguous(),
                            compressor_parts.at(1).narrow(1, token, 1).contiguous(), length,
                            attention_rope);
                        return 0;
                    });
                    return absolute_length / compress_ratio;
                },
                [&] {
                    profiler.measure("dsv4.indexer_compressor_update", [&]() {
                        indexer_compressor.update(
                            state.indexer_compressor, indexer_parts.at(0).narrow(1, token, 1).contiguous(),
                            indexer_parts.at(1).narrow(1, token, 1).contiguous(), length,
                            attention_rope);
                        return 0;
                    });
                    return int64_t{0};
                },
                [&](int64_t visible) {
                    auto int_options = ::tyloi::TensorOptions()
                                           .device(::tyloi::kCUDA)
                                           .dtype(::tyloi::kInt32);
                    return tyloi::models::deepseek_v4::select_compressed(
                        compress_ratio, visible,
                        [&] {
                            auto qr_token = qr.reshape({batch, tokens, qr.size(-1)})
                                                .narrow(1, token, 1)
                                                .reshape({batch, qr.size(-1)})
                                                .contiguous();
                            auto index_query = profiler.measure("dsv4.indexer_q", [&]() {
                                return project.run(*indexer_q, qr_token)
                                    .reshape({batch, 1, heads, 128})
                                    .transpose(1, 2)
                                    .contiguous();
                            });
                            index_query = attention_rope.forward(index_query, position, false)
                                              .transpose(1, 2)
                                              .contiguous();
                            index_query = nepq_hadamard_input_cuda(
                                              index_query.reshape({batch * heads, 128})
                                                  .to(::tyloi::kFloat16)
                                                  .contiguous(),
                                              scratch.hadamard_signs, 128)
                                              .reshape({batch, 1, heads, 128});
                            index_query = dsv4_fp4_sim_cuda(index_query.contiguous());
                            auto weights = profiler.measure("dsv4.indexer_weight", [&]() {
                                return ::tyloi::cuda::matmul(
                                           x.narrow(1, token, 1)
                                               .reshape({batch, hidden_size})
                                               .to(::tyloi::kFloat32),
                                           indexer_weight.transpose(0, 1))
                                    .reshape({batch, 1, heads})
                                    .to(::tyloi::kFloat16)
                                    .contiguous();
                            });
                            auto scores = profiler.measure("dsv4.indexer_scores", [&]() {
                                return dsv4_indexer_scores_cuda(
                                    index_query,
                                    state.indexer_compressor.pool.narrow(1, 0, visible).contiguous(),
                                    weights, cache_pos + token, 4);
                            });
                            return profiler.measure("dsv4.indexer_topk", [&]() {
                                return dsv4_topk512_cuda(scores);
                            });
                        },
                        [&] {
                            return ::tyloi::cuda::arange(visible, int_options)
                                .reshape({1, 1, visible})
                                .expand({batch, 1, visible})
                                .contiguous();
                        },
                        [&] {
                            return ::tyloi::cuda::arange(visible, int_options)
                                .reshape({1, 1, visible})
                                .expand({batch, 1, visible})
                                .contiguous();
                        });
                },
                [&](DenseTensor selected, int64_t visible) {
                    const int64_t plan_ratio = compress_ratio > 0 ? compress_ratio : 1;
                    return profiler.measure("dsv4.attention_plan", [&]() {
                        return dsv4_build_decode_plan_cuda(selected, length, visible,
                                                           plan_ratio, 128);
                    });
                },
                [&](int64_t visible) {
                    return profiler.measure("dsv4.attention_cache", [&]() {
                        return visible > 0
                                   ? ::tyloi::cuda::cat(
                                         {local_cache, state.compressor.pool.narrow(1, 0, visible)},
                                         1)
                                         .contiguous()
                                   : local_cache;
                    });
                },
                [&](DenseTensor cache, const auto &plan) {
                    auto query = queries.narrow(2, token, 1).contiguous();
                    return profiler.measure("dsv4.sparse_attention", [&]() {
                        return attention_dsv4_sparse_cuda(
                            query, cache, plan.at(0), plan.at(1), sinks,
                            scratch.attention_meta,
                            1.0 / std::sqrt(static_cast<double>(head_dim)));
                    });
                },
                [&](DenseTensor attention) {
                    return profiler.measure("dsv4.attention_inverse_rope", [&]() {
                        auto transposed = attention.transpose(1, 2).contiguous();
                        transposed =
                            attention_rope.forward(transposed, position, true);
                        return transposed.transpose(1, 2).contiguous();
                    });
                },
                [&](DenseTensor attention) { return project.low_rank(attention, output_a, output_b, groups, mode.groupwise_output, true); });
        },
        [](const auto &outputs) { return ::tyloi::cuda::cat(outputs, 1); });
}

DenseTensor forward_layer(const Layer& w, LayerState& state, Executor& executor, DenseTensor x,
                     DenseTensor current_ids, const AttentionStep& step, const Executor::Mode& mode) {
    auto& profiler = executor.profiler();
    const auto& hidden_size = w.hidden_size;
    const auto& eps = w.eps;
    const auto& attn_norm = w.attn_norm;
    const auto& ffn_norm = w.ffn_norm;
    const auto& hc_attn_fn = w.hc_attn_fn;
    const auto& hc_attn_scale = w.hc_attn_scale;
    const auto& hc_attn_base = w.hc_attn_base;
    const auto& hc_ffn_fn = w.hc_ffn_fn;
    const auto& hc_ffn_scale = w.hc_ffn_scale;
    const auto& hc_ffn_base = w.hc_ffn_base;
    const auto& ffn = w.ffn;
    if (!current_ids.defined()) {
        throw std::runtime_error("DeepSeek V4 block did not receive token ids");
    }
    const int64_t batch = x.size(0);
    const int64_t tokens = x.size(1);
    return tyloi::models::hyperconnection_layer(
        std::move(x),
        [&](::tyloi::DenseTensor x) {
            auto pre = profiler.measure("dsv4.hc_attn_pre", [&]() {
                return hc_pre(w, x, hc_attn_fn, hc_attn_scale, hc_attn_base, "attn_pre");
            });
            auto normalized = profiler.measure("dsv4.attn_norm", [&]() {
                return rms_norm_cuda(pre.at(0)
                                         .reshape({batch * tokens, hidden_size})
                                         .to(::tyloi::kFloat32),
                                     attn_norm, eps)
                    .reshape({batch, tokens, hidden_size})
                    .to(::tyloi::kFloat16);
            });
            return std::make_pair(normalized, pre);
        },
        [&](const auto &mix) {
            return attention_forward(w, state, executor, mix.first, step, mode);
        },
        [&](::tyloi::DenseTensor attention, ::tyloi::DenseTensor residual,
            const auto &mix) {
            return profiler.measure("dsv4.hc_attn_post", [&]() {
                return hc_post(w, attention, residual, mix.second.at(1), mix.second.at(2),
                               "attn_post");
            });
        },
        [&](::tyloi::DenseTensor x, const auto &) {
            auto pre = profiler.measure("dsv4.hc_ffn_pre", [&]() {
                return hc_pre(w, x, hc_ffn_fn, hc_ffn_scale, hc_ffn_base, "ffn_pre");
            });
            auto normalized = profiler.measure("dsv4.ffn_norm", [&]() {
                return rms_norm_cuda(pre.at(0)
                                         .reshape({batch * tokens, hidden_size})
                                         .to(::tyloi::kFloat32),
                                     ffn_norm, eps)
                    .reshape({batch, tokens, hidden_size})
                    .to(::tyloi::kFloat16);
            });
            return std::make_pair(normalized, pre);
        },
        [&](const auto &mix) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            auto feed_forward = ffn.forward(
                mix.first.reshape({batch * tokens, hidden_size}), mode.ffn, current_ids);
#else
            auto feed_forward = ffn.forward(
                mix.first.reshape({batch * tokens, hidden_size}), executor, mode.ffn, current_ids);
#endif
            feed_forward = feed_forward.reshape({batch, tokens, hidden_size});
            return feed_forward;
        },
        [&](::tyloi::DenseTensor feed_forward, ::tyloi::DenseTensor residual,
            const auto &mix) {
            return profiler.measure("dsv4.hc_ffn_post", [&]() {
                return hc_post(w, feed_forward, residual, mix.second.at(1), mix.second.at(2),
                               "ffn_post");
            });
        },
        [](const auto &) {});
}
} // namespace tyloi::cuda::deepseek_v4
