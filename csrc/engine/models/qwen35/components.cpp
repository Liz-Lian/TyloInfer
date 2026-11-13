#include "engine/models/source.h"
#include "engine/ops.h"
#include "engine/models/qwen35/components.h"

#include "engine/models/common/load.h"
#include "engine/weights/loader.h"
#include "engine/models/common/vision/grid_encoder.h"
#include "engine/models/qwen35/forward.h"
#include "engine/models/qwen35/load.h"
#include "engine/models/qwen35/mtp.h"
#include "engine/models/qwen35/state.h"
#include "engine/backends/cuda/kernels/mfq_cuda_kv_quantization.h"

#include "engine/weights/placement.h"

#include <iostream>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda::qwen35 {

tyloi::engine::Metrics Components::metrics() const {
    const auto component_state = state();
    const auto quantization = cuda_kv_quantization();
    return {
        {"kv_memory_supported", 0.0},
        {"kv_quantization_supported", 1.0},
        {"kv_quantization_bits", quantization.bits},
        {"vision_declared", double(component_state.vision_declared)},
        {"vision_supported", double(component_state.vision_supported)},
        {"vision_available", double(component_state.vision_available)},
        {"mtp_declared", double(component_state.mtp_declared)},
        {"mtp_supported", double(component_state.mtp_supported)},
        {"mtp_available", double(component_state.mtp_available)}};
}

void Components::describe(tyloi::engine::EngineInfo& info) const {
    const auto component_state = state();
    info.multimodal = component_state.vision_available;
    const auto caps = ::tyloi::models::runtime_capabilities(
        graph, plan, component_state, info.multimodal, false);
    info.duplex = caps.full_duplex;
    info.capabilities = {graph.architecture, caps.text,
        caps.image_input, caps.video_input, caps.audio_input, caps.audio_output,
        caps.full_duplex, caps.mtp, "model-graph+cuda-adapters"};
}

tyloi::StepSequence<tyloi::models::Prepared> Components::prepare(
    const Model& language, tyloi::cache::Media& images,
    int primary,
    const std::vector<int64_t>& tokens, const MfqMultimodalInput& media)
    const
    {
    if (grid_vision)
        return grid_vision->prepare(language, images,
            primary,
            tokens, media);
    throw std::invalid_argument("model has no multimodal component");
}

Components Components::load(
    Model& model,
    const ::tyloi::weights::Loader& weights,
    LayerPlacementConfig& placement,
    int cpu_layers,
    bool optional,
    bool cpu_offload,
    bool transfer_auto) {
    Components components{model.graph, model.plan};
    if (!optional) return components;
    const bool mtp_allowed =
        !placement.enabled() && cpu_layers == 0 && !cpu_offload &&
        transfer_auto;

    if (components.plan.vision == ::tyloi::models::Vision::grid_vit) {
        const auto* component = components.graph.component("vision");
        const auto& config = model.config;
        if (component == nullptr || !config.grid_vision ||
            !config.image_token_id || !config.video_token_id) {
            throw std::runtime_error(
                "CUDA grid-Vision configuration is incomplete");
        }
        components.grid_vision.emplace(
            grid_vision_runtime::CudaGridVisionPromptComponent::load(
                weights,
                *config.grid_vision,
                *config.image_token_id, *config.video_token_id,
                component->input_contract, component->position_policy));
    } else if (components.plan.vision != ::tyloi::models::Vision::none) {
        throw std::runtime_error("CUDA vision adapter is unsupported for Qwen");
    }

    if (components.plan.predictor == ::tyloi::models::Predictor::qwen35) {
        if (mtp_allowed && model.metadata.num_experts == 0 &&
            supports_speculation(model)) {
            TransformerWeightLoader loader{weights};
            auto predictor = load_predictor(
                loader, model.config, cuda_kv_quantization());
            if (predictor) {
                components.mtp = std::make_unique<Mtp>(
                    std::move(*predictor), model.embed, model.rope,
                    weights.placement.device);
            }
        } else {
            std::cerr
                << "qwen_mtp unavailable: CUDA adapter requires dense GPU-resident Qwen blocks\n";
        }
    } else if (components.plan.predictor != ::tyloi::models::Predictor::none) {
        throw std::runtime_error("CUDA predictor adapter is unsupported for Qwen");
    }
    return components;
}

} // namespace tyloi::cuda::qwen35
