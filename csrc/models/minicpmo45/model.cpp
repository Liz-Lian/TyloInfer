#include "models/minicpmo45/model.h"
#include "backends/cuda/engine/models/load.h"

#include <iostream>
#include <stdexcept>
#include <utility>

namespace mfq::cuda::minicpmo45 {

std::array<Task, 2> Layer::tasks() const {
    return {{
        {task::Attention{0, 1, 2, attention, 0}, attention.layer},
        {task::FullFfn{1, 2, 0, ffn, ffn_norm, attention}, attention.layer},
    }};
}

Model read(std::shared_ptr<const mfq::ModelSource> source, std::string_view definition, int64_t context) {
    validate_model_source(*source);
    Model model;
    model.source = std::move(source);
    model.graph = model.source->resolved_model_graph();
    if (model.graph.backbone != "minicpmo_tts")
        throw std::runtime_error("MiniCPM TTS loading requires the minicpmo_tts backbone");
    model.plan = cuda_model_plan(model.graph);
    auto& config = model.config;
    config = mfq::models::ModelConfig::from_json(definition);
    if (config.num_hidden_layers != model.graph.topology.text_layers)
        throw std::runtime_error("model graph/config text-layer topology mismatch");
    if (context > 0) {
        if (context > config.max_position_embeddings)
            throw std::runtime_error("--ctx-size exceeds max_position_embeddings");
        config.max_position_embeddings = context;
    }
    for (const auto& type : config.layer_types)
        if (type != "full_attention") throw std::runtime_error("unsupported MiniCPM layer type: " + type);
    auto& metadata = model.metadata;
    metadata.vocab_size = config.vocab_size;
    metadata.hidden_size = config.hidden_size;
    metadata.num_hidden_layers = config.num_hidden_layers;
    metadata.num_attention_heads = config.num_attention_heads;
    metadata.num_key_value_heads = config.num_key_value_heads;
    metadata.head_dim = config.head_dim;
    metadata.max_position_embeddings = config.max_position_embeddings;
    metadata.rotary_dim = config.rotary_dim;
    metadata.rope_base = config.rope_base;
    metadata.rms_norm_eps = config.rms_norm_eps;
    metadata.norm_weight_offset = 1.0;
    metadata.tie_word_embeddings = config.tie_word_embeddings;
    metadata.model_type = config.model_type;
    metadata.layer_types = config.layer_types;
    return model;
}

void load(Model& model, const weight_loader::Loader& weights, const LayerPlacementConfig& placement,
          int cpu_layers, const mfq::models::YarnScaling& yarn) {
    namespace tb = mfq_tensor_backend;
    const auto& config = model.config;
    MfqCudaGuard guard(weights.placement.device);
    const auto make_rope = [&](tb::Device device) {
        return RopeCache(config.max_position_embeddings, config.rotary_dim, config.rope_base,
            0, -1, device, false, yarn);
    };
    model.rope = make_rope(tb::Device(tb::kCUDA, weights.placement.device));
    if (cpu_layers > 0) model.cpu_rope = make_rope(tb::Device(tb::kCPU));
    if (placement.enabled())
        for (int device : placement.devices) {
            MfqCudaGuard device_guard(device);
            model.device_ropes.emplace(device, make_rope(tb::Device(tb::kCUDA, device)));
        }
    const bool separate = !config.tie_word_embeddings && weights.has("model.output.weight");
    model.embed = weights.embedding("model.token_embedding.weight", separate);
    model.output_norm = weights.tensor("model.output_norm.weight");
    model.lm_head = separate ? weights.projection("model.output.weight", TensorParallelAxis::Output)
        : weights.placement.tensor.enabled()
            ? weights.projection("model.token_embedding.weight", TensorParallelAxis::Output) : model.embed;
    model.layers.reserve(config.num_hidden_layers);
    for (int index = 0; index < config.num_hidden_layers; ++index) {
        auto& layer = model.layers.emplace_back();
        layer.device = placement.device_for_layer(index);
        layer.cpu = index < cpu_layers;
        MfqCudaGuard layer_guard(layer.device);
        std::cerr << "loading layer " << index << " full_attention " << (layer.cpu ? "CPU" : "CUDA") << '\n';
        const auto placed = weights.for_layer(layer.device, layer.cpu);
        TransformerWeightLoader loader{placed};
        const auto prefix = "model.block." + std::to_string(index) + ".";
        mfq::models::load_attention_mixer(layer.attention, loader, config, index, prefix, 1.0);
        layer.ffn_norm = loader.dense(prefix + "mlp.norm.weight");
        layer.ffn = mfq::models::load_dense_ffn<Ffn>(loader, config, prefix + "mlp.");
    }
}

} // namespace mfq::cuda::minicpmo45
