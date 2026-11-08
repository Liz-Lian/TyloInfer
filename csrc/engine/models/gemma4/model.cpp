#include "engine/models/gemma4/model.h"
#include "engine/models/common/check.h"
#include "engine/models/common/load.h"

#include <bit>
#include <cmath>
#include <utility>

namespace tyloi::cuda::gemma4 {

tyloi::models::Info Model::info() const {
    bool decode_graph = true;
    for (const auto& layer : layers) {
        if (layer.gemma4_moe && (layer.gemma_moe_gate_up->device().is_cpu() ||
                                 layer.gemma_moe_down->device().is_cpu())) {
            decode_graph = false;
            break;
        }
    }
    return {config.vocab_size, config.max_position_embeddings, config.model_type,
            std::size_t(config.hidden_size) * sizeof(float) * 8, decode_graph, true};
}

MfqModelCapabilities Model::capabilities() const {
    return {graph.architecture, graph.has_component("text"), false, false,
        false, false, false, false, "model-graph"};
}

tyloi::engine::Metrics Model::metrics() const {
    return {{"kv_memory_supported", 0.0},
        {"kv_quantization_supported", 0.0},
        {"kv_quantization_bits", 0.0},
        {"vision_declared", double(graph.has_component("vision"))},
        {"vision_supported", 0.0},
        {"vision_available", 0.0},
        {"mtp_declared", double(graph.has_component("predictor"))},
        {"mtp_supported", 0.0},
        {"mtp_available", 0.0}};
}

Model read(std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
           std::int64_t context) {
    validate_model_source(*source);
    Model model;
    model.source = std::move(source);
    model.graph = tyloi::models::graph(*model.source);
    tyloi::models::require_model(model.graph.backbone == "gemma4",
                               "model backbone does not match the requested causal model");
    model.config = Config::from_json(definition);
    auto bits = std::bit_cast<std::uint32_t>(
        static_cast<float>(std::sqrt(static_cast<double>(model.config.hidden_size))));
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    model.embed_scale = std::bit_cast<float>(bits & 0xffff0000U);
    tyloi::models::require_model(model.config.num_hidden_layers == model.graph.topology.text_layers,
                               "model graph/config text-layer topology mismatch");
    if (context > 0) {
        tyloi::models::require_model(context <= model.config.max_position_embeddings,
                                   "--ctx-size exceeds max_position_embeddings");
        model.config.max_position_embeddings = context;
    }
    return model;
}

} // namespace tyloi::cuda::gemma4
