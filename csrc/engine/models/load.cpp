#include "engine/models/load.h"

#include "engine/models/causal_model.h"
#include "engine/models/gemma4/load.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/models/glm_dsa/load.h"
#include "engine/models/deepseek_v4/load.h"
#include "engine/models/glm5_next/load.h"
#include "engine/models/nemotron_h/load.h"
#include "engine/models/deepseek_v41/load.h"
#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/qwen35/load.h"
#include "engine/models/minicpmo45/model.h"
#include "engine/models/minicpmo45/text/load.h"
#endif
#include "engine/models/source.h"
#include "engine/models/source/model_graph.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace tyloi::models {
Backbone backbone(
        std::string_view implementation) noexcept {
    if (implementation == "qwen3_5" ||
        implementation == "generic_qwen") {
        return Backbone::generic_qwen;
    }
    if (implementation == "minicpmo45") {
        return Backbone::minicpmo45;
    }
    if (implementation == "minicpmo_tts") {
        return Backbone::minicpmo_tts;
    }
    if (implementation == "gemma4") return Backbone::gemma4;
    if (implementation == "glm_dsa") return Backbone::glm_dsa;
    if (implementation == "glm5_next") return Backbone::glm5_next;
    if (implementation == "qwen4_exp") return Backbone::qwen4_exp;
    if (implementation == "deepseek_v4") {
        return Backbone::deepseek_v4;
    }
    if (implementation == "deepseek_v41") {
        return Backbone::deepseek_v41;
    }
    if (implementation == "nemotron_h") {
        return Backbone::nemotron_h;
    }
    return Backbone::unsupported;
}

Plan model_plan(
        const ModelGraph& graph) noexcept {
    Plan result;
    result.backbone = backbone(graph.backbone);
    const auto* vision = graph.component("vision");
    if ((graph.backbone == "qwen3_5" || graph.backbone == "qwen4_exp") && vision != nullptr &&
        vision->tensor_root == "vision" &&
        vision->implementation == "grid_vit" &&
        vision->input_contract == kGridVisionInputContract &&
        vision->position_policy == kGridMropePositionPolicy) {
        result.vision = Vision::grid_vit;
    }
    const auto* audio = graph.component("audio_input");
    if(result.backbone==Backbone::deepseek_v41 && vision && vision->tensor_root=="vision" &&
            vision->implementation=="deepseek_v41_vision" &&
            vision->input_contract=="deepseek_v41_vision.v1" &&
            vision->position_policy=="deepseek_v41_positions")result.vision=Vision::deepseek_v41;
    const auto* tts = graph.component("audio_output");
    // The current CUDA MiniCPM adapter is one composite implementation.  Do
    // not advertise a partially declared graph as supported until those
    // weighted components have independent adapters.
    if (result.backbone == Backbone::minicpmo45 &&
        vision != nullptr && audio != nullptr && tts != nullptr &&
        vision->implementation == "minicpmo45_vision" &&
        audio->implementation == "minicpmo45_audio" &&
        tts->implementation == "minicpmo45_tts") {
        result.vision = Vision::minicpmo45;
    }
    // Cropped-position-embedding towers are selected by their declared data
    // contract alone: the graph names the implementation and the patch
    // contract, and RADIO's graph carries no multimodal position policy.
    if (vision != nullptr && vision->tensor_root == "vision" &&
        vision->implementation == "radio" &&
        vision->input_contract == kRadioCpeInputContract &&
        vision->position_policy.empty()) {
        result.vision = Vision::radio;
    }
    const auto* predictor = graph.component("predictor");
    if (graph.backbone == "qwen3_5" && predictor != nullptr &&
        predictor->implementation == "next_token_prediction" &&
        predictor->tensor_root == "predictor") {
        result.predictor = Predictor::qwen35;
    }
    if ((graph.backbone=="qwen4_exp" || graph.backbone=="glm5_next") && predictor!=nullptr &&
        predictor->implementation=="next_token_prediction" && predictor->tensor_root=="predictor") {
        result.predictor=Predictor::flash_next;
    }
    if (graph.backbone == "deepseek_v41" && predictor != nullptr &&
        predictor->implementation == "deepseek_v41_dspark" &&
        predictor->tensor_root == "predictor") {
        result.predictor = Predictor::deepseek_v41_dspark;
    }
    if (graph.backbone == "nemotron_h" && predictor != nullptr &&
        predictor->implementation == "nemotron_h_mtp" &&
        predictor->tensor_root == "predictor") {
        result.predictor = Predictor::nemotron_h;
    }
    return result;
}

Capabilities capability_matrix(
        const Plan& plan) noexcept {
    Capabilities result;
    result.text = plan.backbone != Backbone::unsupported;
    if (!result.text) return result;
    result.image_input =
        plan.vision == Vision::grid_vit ||
        plan.vision == Vision::minicpmo45 ||
        plan.vision == Vision::deepseek_v41 ||
        plan.vision == Vision::radio;
    result.video_input = plan.vision == Vision::minicpmo45;
    result.audio_input = plan.vision == Vision::minicpmo45;
    result.audio_output = plan.vision == Vision::minicpmo45;
    result.full_duplex = plan.vision == Vision::minicpmo45;
    result.mtp = plan.predictor != Predictor::none;
    return result;
}

ComponentState component_state(
        const ModelGraph& graph,
        const Plan& plan,
        bool vision_loaded,
        bool mtp_loaded,
        bool enable_vision,
        bool enable_mtp) noexcept {
    ComponentState result;
    result.vision_declared = graph.has_component("vision");
    result.vision_supported = result.vision_declared &&
        plan.vision != Vision::none;
    result.vision_available = result.vision_supported && vision_loaded;
    result.vision_enabled = result.vision_available && enable_vision;
    result.mtp_declared = graph.has_component("predictor");
    result.mtp_supported = result.mtp_declared &&
        plan.predictor != Predictor::none;
    result.mtp_available = result.mtp_supported && mtp_loaded;
    result.mtp_enabled = result.mtp_available && enable_mtp;
    return result;
}

Capabilities runtime_capabilities(
        const ModelGraph& graph,
        const Plan& plan,
        const ComponentState& components,
        bool multimodal_available,
        bool duplex_available) noexcept {
    auto result = capability_matrix(plan);
    result.text = result.text && graph.has_component("text");
    result.image_input = result.image_input &&
        components.vision_available && multimodal_available;
    result.video_input = result.video_input &&
        components.vision_available && multimodal_available;
    result.audio_input = result.audio_input && multimodal_available &&
        graph.has_component("audio_input");
    result.audio_output = result.audio_output && duplex_available &&
        graph.has_component("audio_output");
    result.full_duplex = result.full_duplex && duplex_available &&
        graph.has_component("duplex");
    result.mtp = result.mtp && components.mtp_available;
    return result;
}

bool supports(std::string_view implementation) {
    switch (backbone(implementation)) {
    case Backbone::gemma4: return true;
#ifdef TYLOINFER_NATIVE_CUDA
    case Backbone::generic_qwen: case Backbone::glm_dsa: case Backbone::glm5_next:
    case Backbone::qwen4_exp: case Backbone::deepseek_v4: case Backbone::deepseek_v41:
    case Backbone::nemotron_h: case Backbone::minicpmo45: case Backbone::minicpmo_tts: return true;
#endif
    default: return false;
    }
}

std::unique_ptr<CausalModel> load(
    std::shared_ptr<const ModelSource> source,
    std::string_view definition,
    const weights::Loader& weights,
    LayerPlacementConfig& placement,
    const LoadOptions& options) {
    const auto model_graph = graph(*source);
    switch (backbone(model_graph.backbone)) {
    case Backbone::gemma4:
        if (!options.cpu_offload_layers.empty())
            throw std::invalid_argument("Gemma4 does not support --cpu-offload-layers");
        return ::tyloi::cuda::gemma4::load(
            std::move(source), definition, options.context, weights, placement,
            options.gpu_layers, options.expert_parallel);
#ifdef TYLOINFER_NATIVE_CUDA
    case Backbone::glm_dsa:
        return ::tyloi::cuda::glm_dsa::load(std::move(source), definition, weights, placement, options);
    case Backbone::deepseek_v4:
        return ::tyloi::cuda::deepseek_v4::load(std::move(source), definition, weights, placement, options);
    case Backbone::glm5_next:
        return ::tyloi::cuda::glm5_next::load(std::move(source), definition, weights, placement, options);
    case Backbone::nemotron_h:
        return ::tyloi::cuda::nemotron_h::load(std::move(source), definition, weights, placement, options);
    case Backbone::deepseek_v41:
        return ::tyloi::cuda::deepseek_v41::load(std::move(source), definition, weights, placement, options);
    case Backbone::qwen4_exp:
        return ::tyloi::cuda::qwen4_exp::load(std::move(source), definition, weights, placement, options);
    case Backbone::generic_qwen:
        return ::tyloi::cuda::qwen35::load(std::move(source), definition, weights, placement, options);
    case Backbone::minicpmo45:
        return ::tyloi::cuda::minicpmo45::text::load(std::move(source), definition, weights, placement, options);
    case Backbone::minicpmo_tts:
        return ::tyloi::cuda::minicpmo45::load(std::move(source), definition, weights, placement, options);
#endif
    default:
        break;
    }
    throw std::runtime_error(
        "unsupported model backbone: " + model_graph.backbone);
}

} // namespace tyloi::models
