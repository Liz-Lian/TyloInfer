#include "engine/models/qwen4_exp/components.h"

#include "engine/backends/cuda/executor/execution.h"
#include "engine/models/source/model_config.h"
#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/qwen4_exp/predictor.h"
#include "engine/weights/loader.h"

#include <stdexcept>
#include <utility>

namespace tyloi::cuda::qwen4_exp {

tyloi::engine::Metrics Components::profile_metrics(const Model& language, const State& state) const {
    return qwen4_decode_graph_profile_metrics(language, state, mtp.get());
}

std::optional<tyloi::engine::Metrics> Components::memory_metrics(const Model& language, const State& state) const {
    return qwen4_cache_memory_metrics(language, state, mtp.get());
}

tyloi::StepSequence<tyloi::models::Prepared> Components::prepare(
        Model& language, MediaCache& images,
#ifndef TYLOINFER_NATIVE_CUDA
        Projector projector, Collectives& collectives,
#endif
        int primary,
#ifndef TYLOINFER_NATIVE_CUDA
        Executor* executor,
#endif
        const std::vector<int64_t>& tokens, const MfqMultimodalInput& media) {
    if (grid_vision) return grid_vision->prepare(language, images,
#ifndef TYLOINFER_NATIVE_CUDA
        projector, collectives,
#endif
        primary,
#ifndef TYLOINFER_NATIVE_CUDA
        executor,
#endif
        tokens, media);
    throw std::invalid_argument("model has no multimodal component");
}

Components load_components(
        tyloi::cuda::qwen4_exp::Model& model,
        const tyloi::weights::Loader& weights,
        bool load_optional_components) {
    Components result;
    if (!load_optional_components) return result;
    if (model.plan.vision == tyloi::cuda::CudaVisionAdapter::grid_vit) {
        const auto document = nlohmann::json::parse(
            tyloi::models::config_json(*model.source));
        const auto& vision = document.at("vision_config");
        if (vision.value("hidden_act", std::string("gelu_pytorch_tanh")) != "gelu_pytorch_tanh" ||
                !vision.value("deepstack_visual_indexes", std::vector<int>{}).empty())
            throw std::runtime_error("unsupported Qwen4 grid-ViT activation or deepstack");
        tyloi::GridVisionConfig config;
        config.hidden_size = vision.at("hidden_size").get<int64_t>();
        config.intermediate_size = vision.at("intermediate_size").get<int64_t>();
        config.depth = vision.at("depth").get<int64_t>();
        config.num_heads = vision.at("num_heads").get<int64_t>();
        config.in_channels = vision.value("in_channels", int64_t{3});
        config.patch_size = vision.at("patch_size").get<int64_t>();
        config.temporal_patch_size = vision.at("temporal_patch_size").get<int64_t>();
        config.spatial_merge_size = vision.value("spatial_merge_size", int64_t{2});
        config.out_hidden_size = vision.at("out_hidden_size").get<int64_t>();
        config.num_position_embeddings = vision.at("num_position_embeddings").get<int64_t>();
        config.rope_theta = vision.value("rope_theta", 10000.0);
        if (vision.contains("rope_parameters"))
            config.rope_theta = vision.at("rope_parameters").value("rope_theta", config.rope_theta);
        config.layer_norm_eps = vision.value("layer_norm_eps", 1e-6);
        config.validate();
        const auto image = document.at("image_token_id").get<int64_t>();
        const auto video = document.at("video_token_id").get<int64_t>();
        if (config.out_hidden_size != model.config.hidden || model.config.sections.size() != 3 ||
                image < 0 || video < 0 || image >= model.config.vocab || video >= model.config.vocab || image == video)
            throw std::runtime_error("invalid Qwen4 grid-ViT text geometry or placeholder IDs");
        const auto* component = model.graph.component("vision");
        result.grid_vision.emplace(
            tyloi::cuda::grid_vision_runtime::CudaGridVisionPromptComponent::load(
                weights, model.execution->weights, config, image, video,
                component->input_contract, component->position_policy));
    } else if (model.plan.vision != tyloi::cuda::CudaVisionAdapter::none) {
        throw std::runtime_error("unsupported Qwen4 CUDA vision adapter");
    }
    if (model.plan.predictor != tyloi::cuda::CudaPredictorAdapter::none &&
             model.plan.predictor !=
                 tyloi::cuda::CudaPredictorAdapter::flash_next) {
        throw std::runtime_error(
            "unsupported Qwen4 CUDA component adapter");
    }
    if (model.plan.predictor ==
            tyloi::cuda::CudaPredictorAdapter::none) return result;
    TYLOINFER_RUNTIME_CHECK(
        model.supports_speculation(),
        "invalid Flash-Next predictor model");
    auto predictor = tyloi::cuda::qwen4_exp::Predictor::load_if_present(
        *model.execution, weights, model.config, model.embed, model.lm_head,
        model.metadata.final_logit_softcapping);
    if (predictor) {
        result.mtp =
            std::make_unique<tyloi::cuda::qwen4_exp::Predictor>(
                std::move(*predictor));
    }
    return result;
}

} // namespace tyloi::cuda::qwen4_exp
