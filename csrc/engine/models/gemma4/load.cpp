#include "engine/models/gemma4/load.h"
#include "engine/models/gemma4/model.h"
#include "engine/models/gemma4/weights.h"
#include "engine/ops.h"
#include "engine/weights/loader.h"
#include "engine/weights/tq/source.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda::gemma4 {
struct LayerLoader {
    const tyloi::weights::Loader& weights;
    auto dense(const std::string& name) const {
        return weights.tensor(name);
    }
    void qkv(FullAttention& attention, const std::vector<std::string>& names) const {
        attention.qkv = weights.group(names, 2);
    }
    void output(FullAttention& attention, const std::string& name) const {
        attention.o = weights.weight(name);
    }
    static auto shape(const ::tyloi::Tensor& value) {
        std::vector<int64_t> result;
        if (value) for (int64_t i = 0; i < value.dim(); ++i) result.push_back(value.size(i));
        return result;
    }
    static int64_t elements(const ::tyloi::Tensor& value) {
        return value ? value.numel() : 0;
    }
    auto rope(int64_t capacity, const tyloi::models::gemma4::LayerSpec &spec) {
        return tyloi::ops::rope_table(capacity, spec.head_dim, spec.rope_base,
            tyloi::Device(tyloi::kCUDA, weights.placement.device),
            spec.head_dim, spec.rotary_pairs);
    }
    auto ones(int64_t width) {
        const std::array<int64_t, 1> shape{width};
        auto value = tyloi::ops::full(shape, 1, tyloi::Device(
            weights.placement.cpu ? tyloi::kCPU : tyloi::kCUDA, weights.placement.device));
        return tyloi::ops::cast(value, tyloi::kFloat32);
    }
    auto layer_scale(const std::string &name) {
        auto value = tyloi::ops::cast(weights.tensor(name), tyloi::kFloat16);
        return tyloi::ops::contiguous(value);
    }
    auto router_parameter(const std::string &name) {
        auto value = tyloi::ops::cast(weights.tensor(name), tyloi::kFloat32);
        return tyloi::ops::contiguous(value);
    }
    auto normalized_router_parameter(const std::string &name, double divisor) {
        auto value = router_parameter(name);
        auto divided = tyloi::ops::divide(value, divisor);
        return tyloi::ops::contiguous(divided);
    }
    auto mlp(const std::vector<std::string>& names, const std::string& down_name) {
        auto down = weights.weight(down_name);
        auto group = weights.gate_up(names, down, 2);
        return tyloi::models::Mlp{std::move(group), std::move(down),
                                  tyloi::ops::MlpActivation::gelu_gated, 0.0};
    }
    auto experts(const std::string &name, int layer, const char *projection) {
        if (weights.placement.experts.enabled()) {
            if (tyloi::is_tq_table(weights.source, name))
                throw std::runtime_error("Gemma4 expert parallelism does not support TQ expert banks: " + name);
            const auto& dtype = tyloi::require_tensor(weights.source, name).dtype;
            if (dtype == "F32" || dtype == "F16" || dtype == "BF16")
                throw std::runtime_error("Gemma4 expert parallelism does not support " + dtype + " expert banks: " + name);
        }
        return weights.experts(name, true, layer, projection);
    }
    static auto expert_shape(const tyloi::Tensor& weight) {
        return std::array<int64_t, 3>{weight.size(0), weight.size(1), weight.size(2)};
    }
};

Layer load_layer(const tyloi::weights::Loader& weights, const Config &config,
                 int layer, const std::string &type) {
    Layer result;
    result.device = weights.placement.device;
    LayerLoader loader{weights};
    tyloi::models::gemma4::load_layer(result.attention, result, loader, config, layer, type);
    return result;
}

void load(Model& model, const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
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
    if (load_input_output) {
        const bool separate = !config.tie_word_embeddings &&
            (weights.has("model.output.weight") || tyloi::is_tq_table(weights.source, "model.output.weight"));
        model.embed = weights.embedding("model.token_embedding.weight", separate);
        model.output_norm = weights.tensor("model.output_norm.weight");
        if (separate || weights.placement.tensor.enabled()) {
            model.lm_head = weights.weight(
                separate ? "model.output.weight" : "model.token_embedding.weight",
                TensorParallelAxis::Output);
        } else {
            model.lm_head = model.embed;
        }
        if (load_layers) {
            model.layers.reserve(static_cast<std::size_t>(config.num_hidden_layers));
            for (int index = 0; index < config.num_hidden_layers; ++index) {
                const int device = placement.device_for_layer(index);
                const auto& type = config.layer_types.at(static_cast<std::size_t>(index));
                std::cerr << "loading layer " << index << ' ' << type << " CUDA" << std::endl;
                model.layers.push_back(load_layer(weights.for_layer(device, false), config, index, type));
            }
        }
    } else {
        if (load_layers) throw std::invalid_argument("streamed model setup requires deferred blocks");
        model.output_norm = weights.tensor("model.output_norm.weight");
    }
}

void load(std::unique_ptr<tyloi::models::CausalModel>& owner,
        std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
        std::int64_t context, const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
        int gpu_layers, bool expert_parallel) {
    owner = std::make_unique<Model>(read(std::move(source), definition, context));
    auto* model = static_cast<Model*>(owner.get());
    load(*model, weights, placement, gpu_layers, expert_parallel);
    if (const auto* vision = model->graph.component("vision"); vision != nullptr &&
        vision->tensor_root == "vision" && vision->implementation == "radio" &&
        vision->input_contract == tyloi::kRadioCpeInputContract &&
        vision->position_policy.empty()) {
        throw std::runtime_error("CUDA component adapter is unsupported for this model");
    }
}

} // namespace tyloi::cuda::gemma4
