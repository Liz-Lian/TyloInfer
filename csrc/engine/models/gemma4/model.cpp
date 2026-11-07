#include "engine/models/gemma4/model.h"
#include "engine/models/common/check.h"
#include "engine/models/common/load.h"

#include <bit>
#include <cmath>
#include <utility>

namespace tyloi::cuda::gemma4 {

tyloi::models::Info Model::info() const {
    bool decode_graph = true;
#ifdef TYLOINFER_NATIVE_CUDA
    for (const auto& layer : layers) {
        if (layer.gemma4_moe && (layer.gemma_moe_gate_up->device().is_cpu() ||
                                 layer.gemma_moe_down->device().is_cpu())) {
            decode_graph = false;
            break;
        }
    }
#endif
    return {config.vocab_size, config.max_position_embeddings, config.model_type,
            std::size_t(config.hidden_size) * sizeof(float) * 8, decode_graph, true};
}

MfqModelCapabilities Model::capabilities() const {
    const auto state = cuda_component_state(graph, plan, false, false);
    const auto caps = cuda_runtime_capabilities(graph, plan, state, false, false);
    return {graph.architecture, caps.text, caps.image_input, caps.video_input,
        caps.audio_input, caps.audio_output, caps.full_duplex, caps.mtp,
        "model-graph+cuda-adapters"};
}

tyloi::engine::Metrics Model::metrics() const {
    const auto state = cuda_component_state(graph, plan, false, false);
    return {{"kv_memory_supported", 0.0},
        {"kv_quantization_supported", 0.0},
        {"kv_quantization_bits", 0.0},
        {"vision_declared", double(state.vision_declared)},
        {"vision_supported", double(state.vision_supported)},
        {"vision_available", double(state.vision_available)},
        {"mtp_declared", double(state.mtp_declared)},
        {"mtp_supported", double(state.mtp_supported)},
        {"mtp_available", double(state.mtp_available)}};
}

Model read(std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
           std::int64_t context) {
    validate_model_source(*source);
    Model model;
    model.source = std::move(source);
    model.graph = tyloi::models::graph(*model.source);
    model.plan = cuda_model_plan(model.graph);
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
