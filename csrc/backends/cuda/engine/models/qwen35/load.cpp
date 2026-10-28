#include "load.h"

#include "engine/models/qwen35/metadata.h"
#include "engine/models/qwen35/weights.h"

#include "backends/cuda/engine/executor/execution.h"
#include "backends/cuda/engine/models/load.h"
#include "backends/cuda/engine/weights/load.h"
#include "backends/cuda/engine/weights/moe/expert_cache.h"
#include "engine/weights/group.h"
#include "mfq_cuda_kv_quantization.h"
#include "source/read.h"

#include <algorithm>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <utility>

namespace mfq::cuda::qwen35 {
namespace {

struct LayerLoader : TransformerWeightLoader {
    using Ffn = mfq::cuda::Ffn;
    bool mirror_scalars = true;
    using Tensor = mfq_tensor_backend::Tensor;

    static Tensor log_negative(Tensor value) {
        return mfq_tensor_backend::log(-value);
    }

    void linear_projections(LinearAttention& attention,
                            const mfq::models::qwen35::LinearWeightNames& names,
                            bool split_qkv) const {
        if (split_qkv) {
            attention.split_in_proj = true;
            attention.qkv_proj = weights.group({names.qk, names.value});
            if (is_quant_dtype(require_tensor(weights.source, names.gate).dtype)) {
                attention.z_proj = weights.projection(names.gate);
                const bool alpha_quant = is_quant_dtype(require_tensor(weights.source, names.alpha).dtype);
                const bool beta_quant = is_quant_dtype(require_tensor(weights.source, names.beta).dtype);
                if (alpha_quant != beta_quant)
                    throw std::runtime_error(
                        "linear_attn alpha/beta must use the same storage kind");
                attention.ab_is_nint = alpha_quant;
                if (attention.ab_is_nint) {
                    const auto scalar_axis =
                        mirror_scalars
                            ? std::optional<TensorParallelAxis>(TensorParallelAxis::Mirrored)
                            : std::nullopt;
                    attention.ab_nint_proj = make_quant_group(
                        {weights.projection(names.alpha, scalar_axis),
                         weights.projection(names.beta, scalar_axis)},
                        weights.options.grouped,
                        weights.placement.cpu);
                } else {
                    attention.ab_proj = make_dense_group({
                        weights.tensor(names.alpha),
                        weights.tensor(names.beta)});
                }
            } else {
                attention.split_dense_zab = true;
                attention.zab_proj = make_dense_group({
                    weights.tensor(names.gate),
                    weights.tensor(names.alpha),
                    weights.tensor(names.beta)});
            }
            return;
        }

        const bool alpha_quant = is_quant_dtype(require_tensor(weights.source, names.alpha).dtype);
        const bool beta_quant = is_quant_dtype(require_tensor(weights.source, names.beta).dtype);
        if (alpha_quant != beta_quant)
            throw std::runtime_error(
                "linear_attention alpha/beta must use the same storage kind");
        if (alpha_quant) {
            attention.in_proj = weights.group({names.qkv, names.gate, names.alpha, names.beta});
        } else {
            attention.dense_ab_tail = true;
            attention.in_proj = weights.group({names.qkv, names.gate});
            attention.ab_proj = make_dense_group({
                weights.tensor(names.alpha),
                weights.tensor(names.beta)});
        }
    }

    void linear_output(LinearAttention& attention, const std::string& name) const {
        if (is_quant_dtype(require_tensor(weights.source, name).dtype)) {
            attention.out_proj = weights.projection(name);
        } else {
            attention.dense_out_proj = true;
            attention.out_proj_dense = weights.tensor(name);
            if (require_tensor(weights.source, name).dtype == "F16")
                attention.out_proj_dense =
                    attention.out_proj_dense.to(mfq_tensor_backend::kFloat16).contiguous();
            if (attention.out_proj_dense.dim() != 2)
                throw std::runtime_error(
                    "dense linear_attention output projection must be 2D");
        }
    }
};

} // namespace

Layer load_layer(const weight_loader::Loader& weights, const Config& config,
                 int index, const std::string& type, bool mirror_scalars, std::string_view tensor_root) {
    if (index < 0 || index >= config.num_hidden_layers)
        throw std::out_of_range("Qwen layer index is outside the configured model");
    if (static_cast<std::size_t>(index) < config.layer_types.size() &&
        config.layer_types[static_cast<std::size_t>(index)] != type)
        throw std::invalid_argument("Qwen layer type disagrees with model config");
    MfqCudaGuard scope(weights.placement.device);
    Layer result;
    result.device = weights.placement.device;
    result.cpu = weights.placement.cpu;
    LayerLoader loader{{weights}, mirror_scalars};
    if (Config::attention_kind(type) == Config::AttentionKind::full) {
        Full full;
        full.attention.kv_quantization = cuda_kv_quantization();
        mfq::models::load_attention_mixer(
            full.attention, loader, config, index,
            std::string(tensor_root) + ".block." + std::to_string(index) + ".",
            config.legacy_tensor_layout.norm_weight_offset,
            config.attention_output_gate);
        full.attention.attention_rotary_dim = config.rotary_dim;
        full.attention.no_rope = config.positional_encoding == mfq::models::PositionalEncoding::none;
        full.ffn_norm = loader.dense(std::string(tensor_root) + ".block." +
                                     std::to_string(index) + ".mlp.norm.weight");
        full.ffn = mfq::models::qwen35::load_ffn<Ffn>(loader, config, index, tensor_root);
        result.weights = std::move(full);
    } else {
        Linear linear;
        linear.attention.key_heads = config.linear_num_key_heads;
        linear.attention.value_heads = config.linear_num_value_heads;
        linear.attention.key_dim = config.linear_key_head_dim;
        linear.attention.value_dim = config.linear_value_head_dim;
        linear.attention.conv_kernel = config.linear_conv_kernel_dim;
        linear.attention.rms_norm_eps = config.rms_norm_eps;
        linear.attention.norm_weight_offset = config.legacy_tensor_layout.norm_weight_offset;
        linear.attention.tiled_v_heads = config.legacy_tensor_layout.qwen_gdn_gguf_layout;
        const auto prefix = std::string(tensor_root) + ".block." + std::to_string(index) + ".";
        linear.attention.attn_norm = loader.dense(prefix + "attention.norm.weight");
        linear.ffn_norm = loader.dense(prefix + "mlp.norm.weight");
        mfq::models::qwen35::load_linear_attention(
            linear.attention, loader, config, index, tensor_root);
        linear.ffn = mfq::models::qwen35::load_ffn<Ffn>(loader, config, index, tensor_root);
        result.weights = std::move(linear);
    }
    return result;
}

Model load(CudaExecutionContext& execution, const std::string& model_path,
           const std::string& config_path, std::int64_t context_size,
           bool load_layers, bool defer_cache,
           std::shared_ptr<const mfq::ModelSource> model_source,
           bool load_input_output) {
    Model model;
    model.source = model_source ? std::move(model_source) : mfq::open_model_source(model_path);
    const auto& source = *model.source;
    mfq::cuda::validate_model_source(source);
    model.graph = source.resolved_model_graph();
    model.plan = cuda_model_plan(model.graph);
    if (model.graph.component("vision") != nullptr &&
        model.plan.vision != CudaVisionAdapter::grid_vit)
        throw std::runtime_error("Qwen CUDA vision requires grid_vit/grid_vision.v1/grid_mrope");
    auto definition = mfq::cuda::load_model_config_json(source, config_path);
    definition = mfq::cuda::configure_yarn(execution.yarn, execution.yarn_context_extension, std::move(definition),
                                            model.graph.backbone, context_size);
    if (model.graph.backbone != "qwen3_5" && model.graph.backbone != "generic_qwen")
        throw std::runtime_error("Qwen3.5 loading requires a Qwen-compatible model graph");
    model.config = Config::from_json(definition, model.graph);
    model.config.legacy_tensor_layout = source.tensor_layout();
    if (context_size > 0) {
        if (context_size > model.config.max_position_embeddings)
            throw std::runtime_error("--ctx-size exceeds max_position_embeddings");
        model.config.max_position_embeddings = context_size;
    }
    if (model.config.num_hidden_layers != model.graph.topology.text_layers)
        throw std::runtime_error("model graph/config text-layer topology mismatch");
    model.metadata = mfq::models::qwen35::metadata_from_config(model.config);
    if (execution.expert_parallel.enabled() && model.config.num_experts <= 0)
        throw std::runtime_error("--expert-parallel requires a model with routed experts");

    execution.layer_placement.prepare(model.config.num_hidden_layers);
    execution.dense_cpu_layer_count = 0;
    if (execution.n_gpu_layers >= 0) {
        execution.dense_cpu_layer_count = static_cast<int>(std::max<int64_t>(
            model.config.num_hidden_layers - execution.n_gpu_layers, 0));
        if (execution.dense_cpu_layer_count > 0) {
            if (model_parallel_enabled(execution) || execution.layer_placement.enabled())
                throw std::runtime_error(
                    "--n-gpu-layers cannot be combined with tensor/expert/layer parallelism");
            bool supported = model.config.num_experts <= 0;
            for (const auto& layer_type : model.config.layer_types)
                supported = supported &&
                            (layer_type == "full_attention" || layer_type == "linear_attention");
            if (!supported)
                throw std::runtime_error(
                    "--n-gpu-layers currently supports dense Qwen-style blocks only");
            std::cerr << "dense_layer_placement cpu=0-" << execution.dense_cpu_layer_count - 1
                      << " gpu=" << execution.dense_cpu_layer_count << '-'
                      << model.config.num_hidden_layers - 1 << " cpu_threads="
                      << mfq_get_num_threads() << std::endl;
        }
    }

    CudaScope model_scope(execution);
    if (model.plan.vision == CudaVisionAdapter::grid_vit && execution.dense_cpu_layer_count > 0)
        throw std::runtime_error("CUDA grid-Vision currently requires GPU-resident text layers");

    if (model.config.positional_encoding != mfq::models::PositionalEncoding::none) {
        const auto make_rope = [&](mfq_tensor_backend::Device device) {
            RopeCache rope(model.config.max_position_embeddings, model.config.rotary_dim,
                           model.config.rope_base, 0, -1, device,
                           model.metadata.rope_interleaved, execution.yarn);
            rope.configure_mrope(model.config.mrope_sections, model.config.mrope_interleaved,
                                 model.config.rotary_dim, device);
            return rope;
        };
        const auto primary = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA, execution.layer_placement.primary_device());
        model.rope = make_rope(primary);
        if (execution.dense_cpu_layer_count > 0)
            model.cpu_rope = make_rope(mfq_tensor_backend::Device(mfq_tensor_backend::kCPU));
        if (execution.layer_placement.enabled()) {
            for (int device : execution.layer_placement.devices) {
                MfqCudaGuard rope_guard(device);
                model.device_ropes.emplace(
                    device, make_rope(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, device)));
            }
        }
    }

    weight_loader::Loader weights{source, execution.config.weights,
        {execution.layer_placement.primary_device(), false, execution.drop_file_cache,
         execution.tensor_parallel, moe_parallel_config(execution)},
        execution.stores.weights, execution.tq};
    if (load_input_output) {
        const auto output_name = std::string("model.output.weight");
        const bool independent_output = !model.config.tie_word_embeddings &&
                                        has_tensor(source, output_name);
        model.embed = weights.embedding("model.token_embedding.weight",
                                     independent_output);
        model.output_norm = weights.tensor("model.output_norm.weight");
        if (model.config.tie_word_embeddings || !has_tensor(source, output_name)) {
            model.lm_head = execution.tensor_parallel.enabled()
                ? weights.projection("model.token_embedding.weight",
                                    TensorParallelAxis::Output)
                : model.embed;
        } else {
            model.lm_head = weights.projection(output_name,
                                              TensorParallelAxis::Output);
        }
        if (load_layers) {
            model.layers.reserve(static_cast<std::size_t>(model.config.num_hidden_layers));
            for (int index = 0; index < model.config.num_hidden_layers; ++index) {
                const int device = execution.layer_placement.device_for_layer(index);
                const bool cpu = index < execution.dense_cpu_layer_count;
                std::cerr << "loading layer " << index << ' '
                          << model.config.layer_types.at(static_cast<std::size_t>(index)) << ' '
                          << (cpu ? "CPU" : "CUDA") << std::endl;
                model.layers.push_back(load_layer(
                    weights.for_layer(device, cpu), model.config, index,
                    model.config.layer_types.at(static_cast<std::size_t>(index)),
                    execution.config.tensor_parallel_mirror_linear_attention_scalars));
            }
        }
    } else {
        if (load_layers)
            throw std::invalid_argument("streamed model setup requires deferred layers");
        model.output_norm = weights.tensor("model.output_norm.weight");
    }

    if (moe_expert_cache_has_sources(execution.stores.weights.moe) &&
        !moe_expert_cache_finalized(execution.stores.weights.moe) && !defer_cache)
        finalize_moe_expert_cache(execution.stores.weights.moe);
    return model;
}

std::optional<Predictor> load_predictor(TransformerWeightLoader& loader,
                                         const Config& config,
                                         KvQuantization quantization) {
    const bool fusion_present = loader.has("predictor.fusion.weight");
    const bool any = fusion_present || loader.has("predictor.hidden_norm.weight") ||
        loader.has("predictor.embedding_norm.weight") ||
        loader.has("predictor.output_norm.weight") ||
        loader.has("predictor.block.0.attention.query.weight");
    if (!fusion_present) {
        TYLOINFER_RUNTIME_CHECK(!any, "Qwen model source contains an incomplete MTP head");
        return {};
    }
    TYLOINFER_RUNTIME_CHECK(config.mtp_num_hidden_layers > 0 && !config.mtp_use_dedicated_embeddings,
        "Qwen MTP requires declared predictor layers and shared embeddings");
    Predictor model;
    model.config = config;
    model.hidden_norm = loader.dense("predictor.hidden_norm.weight");
    model.embedding_norm = loader.dense("predictor.embedding_norm.weight");
    model.output_norm = loader.dense("predictor.output_norm.weight");
    model.fusion = loader.linear("predictor.fusion.weight");
    TYLOINFER_RUNTIME_CHECK(model.fusion.neuron_len() == 2 * config.hidden_size &&
        model.fusion.out() == config.hidden_size &&
        model.hidden_norm.numel() == config.hidden_size &&
        model.embedding_norm.numel() == config.hidden_size &&
        model.output_norm.numel() == config.hidden_size,
        "Qwen MTP component dimensions disagree with the backbone");
    model.layers.reserve(config.mtp_num_hidden_layers);
    for (int index = 0; index < config.mtp_num_hidden_layers; ++index) {
        const auto prefix = "predictor.block." + std::to_string(index) + ".";
        Predictor::Layer layer;
        layer.attention.kv_quantization = quantization;
        mfq::models::load_attention_mixer(layer.attention, loader, config, index, prefix,
            config.legacy_tensor_layout.norm_weight_offset, config.attention_output_gate);
        layer.ffn_norm = loader.dense(prefix + "mlp.norm.weight");
        layer.ffn = mfq::models::qwen35::load_ffn<Ffn>(loader, config, index, "predictor");
        model.layers.push_back(std::move(layer));
    }
    return model;
}

} // namespace mfq::cuda::qwen35
