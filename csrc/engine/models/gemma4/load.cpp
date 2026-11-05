#include "engine/models/gemma4/load.h"
#include "engine/models/gemma4/weights.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/weights/loader.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda::gemma4 {
struct LayerLoader {
    const tyloi::weights::Loader& weights;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    auto dense(const std::string& name) const {
        return weights.tensor(name);
    }
#else
    auto dense(const std::string& name) const {
        return dynamic_cast<const tyloi::DenseTensor&>(*weights.tensor(name));
    }
#endif
    void qkv(FullAttention& attention, const std::vector<std::string>& names) const {
        attention.qkv = weights.group(names, 2);
    }
    void output(FullAttention& attention, const std::string& name) const {
        attention.o = weights.weight(name);
    }
    static auto shape(const std::shared_ptr<const tyloi::Tensor>& value) {
        std::vector<int64_t> result;
        if (value) for (int64_t i = 0; i < value->dim(); ++i) result.push_back(value->size(i));
        return result;
    }
    static int64_t elements(const std::shared_ptr<const tyloi::Tensor>& value) {
        return value ? value->numel() : 0;
    }
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
    static auto shape(const tyloi::DenseTensor& value) { return value.sizes().vec(); }
    static int64_t elements(const tyloi::DenseTensor& value) { return value.numel(); }
#endif
    auto rope(int64_t capacity, const tyloi::models::gemma4::LayerSpec &spec) {
        auto cache = RopeCache(capacity, spec.head_dim, spec.rope_base, spec.head_dim, spec.rotary_pairs);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        return cache.view();
#else
        return cache;
#endif
    }
    auto ones(int64_t width) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        const std::array<int64_t, 1> shape{width};
        auto value = tyloi::ops::full(shape, 1, tyloi::Device(
            weights.placement.cpu ? tyloi::kCPU : tyloi::kCUDA, weights.placement.device));
        return tyloi::ops::cast(*value, tyloi::kFloat32);
#else
        return ::tyloi::cuda::ones({width}, ::tyloi::TensorOptions()
            .device(::tyloi::Device(weights.placement.cpu ? tyloi::kCPU : tyloi::kCUDA,
                                    weights.placement.device))
            .dtype(::tyloi::kFloat32));
#endif
    }
    auto layer_scale(const std::string &name) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        auto value = tyloi::ops::cast(*weights.tensor(name), tyloi::kFloat16);
        return tyloi::ops::contiguous(*value);
#else
        return dense(name).to(::tyloi::kFloat16).contiguous();
#endif
    }
    auto router_parameter(const std::string &name) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        auto value = tyloi::ops::cast(*weights.tensor(name), tyloi::kFloat32);
        return tyloi::ops::contiguous(*value);
#else
        return dense(name).to(::tyloi::kFloat32).contiguous();
#endif
    }
    auto normalized_router_parameter(const std::string &name, double divisor) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        auto value = router_parameter(name);
        const auto& dense = dynamic_cast<const tyloi::DenseTensor&>(*value);
        return std::make_shared<tyloi::DenseTensor>((dense / divisor).contiguous());
#else
        return (router_parameter(name) / divisor).contiguous();
#endif
    }
    auto mlp(const std::vector<std::string>& names, const std::string& down_name) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        auto down = weights.weight(down_name);
        auto group = weights.gate_up(names, *down, 2);
        return tyloi::models::Mlp{std::move(group), std::move(down),
                                tyloi::ops::MlpActivation::gelu_gated};
#else
        auto down = weights.weight(down_name);
        auto group = weights.gate_up(names, *down, 2);
        Ffn result;
        result.geglu = true;
        result.down = std::move(down);
        result.gate_up = std::move(group);
        return result;
#endif
    }
    auto experts(const std::string &name, int layer, const char *projection) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if (weights.placement.experts.enabled()) {
            const auto& dtype = tyloi::require_tensor(weights.source, name).dtype;
            if (dtype == "TQ" || dtype == "F32" || dtype == "F16" || dtype == "BF16")
                throw std::runtime_error("Gemma4 expert parallelism does not support " + dtype + " expert banks: " + name);
        }
#endif
        return weights.experts(name, true, layer, projection);
    }
    static auto expert_shape(const tyloi::Tensor& weight) {
        return std::array<int64_t, 3>{weight.size(0), weight.size(1), weight.size(2)};
    }
};

Layer load_layer(const tyloi::weights::Loader& weights, const Config &config,
                 int layer, const std::string &type) {
    MfqCudaGuard guard(weights.placement.device);
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
    MfqCudaGuard guard(placement.primary_device());
    if (load_input_output) {
        const bool separate = !config.tie_word_embeddings && weights.has("model.output.weight");
        model.embed = weights.embedding("model.token_embedding.weight", separate);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        model.output_norm = weights.tensor("model.output_norm.weight");
#else
        model.output_norm = dynamic_cast<const tyloi::DenseTensor&>(*weights.tensor("model.output_norm.weight"));
#endif
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
                MfqCudaGuard layer_guard(device);
                const auto& type = config.layer_types.at(static_cast<std::size_t>(index));
                std::cerr << "loading layer " << index << ' ' << type << " CUDA" << std::endl;
                model.layers.push_back(load_layer(weights.for_layer(device, false), config, index, type));
            }
        }
    } else {
        if (load_layers) throw std::invalid_argument("streamed model setup requires deferred blocks");
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        model.output_norm = weights.tensor("model.output_norm.weight");
#else
        model.output_norm = dynamic_cast<const tyloi::DenseTensor&>(*weights.tensor("model.output_norm.weight"));
#endif
    }
}

} // namespace tyloi::cuda::gemma4
