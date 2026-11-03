#include "engine/models/gemma4/load.h"
#include "engine/models/gemma4/weights.h"
#include "engine/models/common/load.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/weights/loader.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace mfq::cuda::gemma4 {
struct LayerLoader : TransformerWeightLoader {
    auto rope(int64_t capacity, const mfq::models::gemma4::LayerSpec &spec) {
        auto cache = RopeCache(capacity, spec.head_dim, spec.rope_base, spec.head_dim, spec.rotary_pairs);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        return cache.view();
#else
        return cache;
#endif
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
    auto mlp(const std::vector<std::string>& names, const std::string& down_name) {
        auto down = linear(down_name);
        auto group = gate_up(names, down);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        return mfq::models::Mlp{std::make_shared<QuantLinearGroup>(std::move(group)),
                                std::make_shared<QuantLinear>(std::move(down)),
                                mfq::ops::MlpActivation::gelu_gated};
#else
        Ffn result;
        result.geglu = true;
        result.down = std::move(down);
        result.gate_up = std::move(group);
        return result;
#endif
    }
    auto experts(const std::string &name, int layer, const char *projection) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        return std::shared_ptr<const mfq::weights::ExpertBank>(
            std::make_shared<Experts>(weights.experts(name, true, layer, projection)));
#else
        return weights.experts(name, true, layer, projection);
#endif
    }
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    static auto shape(const mfq::Tensor& value) { return value.sizes().vec(); }
    static int64_t elements(const mfq::Tensor& value) { return value.numel(); }
    static auto expert_shape(const std::shared_ptr<const mfq::weights::ExpertBank>& weight) {
        return std::array<int64_t, 3>{weight->count(), weight->rows(), weight->cols()};
    }
#else
    static auto expert_shape(const mfq::cuda::Experts &weight) {
        return std::array<int64_t, 3>{weight.n_experts, weight.out_per_expert, weight.neuron_len};
    }
#endif
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
        model.embed = std::make_shared<QuantLinear>(
            weights.embedding("model.token_embedding.weight", separate));
        model.output_norm = weights.tensor("model.output_norm.weight");
        if (separate || weights.placement.tensor.enabled()) {
            model.lm_head = std::make_shared<QuantLinear>(weights.projection(
                separate ? "model.output.weight" : "model.token_embedding.weight",
                TensorParallelAxis::Output));
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
        model.output_norm = weights.tensor("model.output_norm.weight");
    }
}

} // namespace mfq::cuda::gemma4
