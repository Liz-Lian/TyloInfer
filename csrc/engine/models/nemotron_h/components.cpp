#include "engine/models/nemotron_h/components.h"

#include "engine/models/source/model_config.h"
#include "engine/models/nemotron_h/model.h"
#include "engine/models/nemotron_h/mtp.h"

#include <stdexcept>
#include <utility>

namespace tyloi::cuda::nemotron_h {

Components load_components(Model& model, const tyloi::weights::Loader& weights,
        bool load_optional_components
#ifdef TYLOINFER_NATIVE_CUDA
        ) {
#else
        , CudaExecutionContext& execution) {
#endif
    Components result;
    if (!load_optional_components) return result;
    if ((model.plan.vision != CudaVisionAdapter::none &&
         model.plan.vision != CudaVisionAdapter::radio) ||
            (model.plan.predictor != CudaPredictorAdapter::none &&
             model.plan.predictor != CudaPredictorAdapter::nemotron_h)) {
        throw std::runtime_error("unsupported Nemotron-H CUDA component adapter");
    }
    if (model.plan.vision == CudaVisionAdapter::radio) {
        const auto* component = model.graph.component("vision");
        TYLOINFER_RUNTIME_CHECK(component != nullptr,
            "Nemotron-H graph declares a RADIO adapter without a vision component");
        const auto vision = tyloi::models::nemotron_h::RadioVisionConfig::from_json(
            tyloi::models::config_json(*model.source));
        result.radio_vision.emplace(CudaRadioPromptComponent::load(
            weights, vision, component->input_contract, component->position_policy));
    }
    if (model.plan.predictor == CudaPredictorAdapter::none) return result;
    TYLOINFER_RUNTIME_CHECK(supports_speculation(model), "invalid Nemotron-H predictor model");
    auto predictor = Mtp::load_if_present(
#ifndef TYLOINFER_NATIVE_CUDA
        execution,
#endif
        weights, model.config, model.embed);
    TYLOINFER_RUNTIME_CHECK(predictor,
        "Nemotron-H graph declares the predictor component but the source has no MTP weights");
    result.mtp = std::make_unique<Mtp>(std::move(*predictor));
    return result;
}

tyloi::StepSequence<tyloi::models::Prepared> Components::prepare(Model& language, MediaCache&,
#ifndef TYLOINFER_NATIVE_CUDA
        Projector projector, Collectives& collectives,
#endif
        int primary,
#ifndef TYLOINFER_NATIVE_CUDA
        Executor* executor,
#endif
        const std::vector<int64_t>& tokens, const MfqMultimodalInput& media) {
    if (radio_vision)
        return radio_vision->prepare(language,
#ifndef TYLOINFER_NATIVE_CUDA
            projector, *executor, collectives,
#endif
            primary, tokens, media);
    throw std::invalid_argument("model has no multimodal component");
}

} // namespace tyloi::cuda::nemotron_h
