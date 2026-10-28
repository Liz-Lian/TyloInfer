#include "models/qwen35/components.h"

#include "backends/cuda/engine/models/load.h"
#include "models/common/grid_vision.h"
#include "models/qwen35/forward.h"
#include "models/qwen35/load.h"
#include "models/qwen35/mtp.h"

#include "native/cuda/collectives.h"
#include "engine/executor/run.h"

#include <iostream>
#include <stdexcept>
#include <utility>

namespace mfq::cuda::qwen35 {

RuntimeComponents<Model> load_components(
    const Model& model,
    const weight_loader::Loader& weights,
    const Executor& executor,
    Collectives& copies,
    const int64_t& kv_capacity,
    Trace* const& trace,
    const int& trace_scope,
    bool optional,
    bool mtp_allowed,
    bool preallocate) {
    RuntimeComponents<Model> result{model.graph, model.plan};
    if (!optional) return result;

    if (result.plan.vision == CudaVisionAdapter::grid_vit) {
        const auto* component = result.graph.component("vision");
        const auto& config = model.config;
        if (component == nullptr || !config.grid_vision ||
            !config.image_token_id || !config.video_token_id) {
            throw std::runtime_error(
                "CUDA grid-Vision configuration is incomplete");
        }
        result.grid_vision.emplace(
            grid_vision_runtime::CudaGridVisionPromptComponent::load(
                weights, *config.grid_vision,
                *config.image_token_id, *config.video_token_id,
                component->input_contract, component->position_policy));
        result.vision_available = true;
    } else if (result.plan.vision != CudaVisionAdapter::none) {
        throw std::runtime_error("CUDA vision adapter is unsupported for Qwen");
    }

    if (result.plan.predictor == CudaPredictorAdapter::qwen35) {
        if (mtp_allowed && model.metadata.num_experts == 0 &&
            supports_speculation(model)) {
            TransformerWeightLoader loader{weights};
            auto predictor = load_predictor(
                loader, model.config, cuda_kv_quantization());
            if (predictor) {
                // Mtp retains these references, so they must outlive the result.
                result.mtp = std::make_unique<Mtp>(
                    std::move(*predictor), model.embed, model.rope,
                    executor.fork(), copies, weights.placement.device,
                    kv_capacity, trace, trace_scope, preallocate);
            }
            result.mtp_available = static_cast<bool>(result.mtp);
        } else {
            std::cerr
                << "qwen_mtp unavailable: CUDA adapter requires dense GPU-resident Qwen blocks\n";
        }
    } else if (result.plan.predictor != CudaPredictorAdapter::none) {
        throw std::runtime_error("CUDA predictor adapter is unsupported for Qwen");
    }
    return result;
}

} // namespace mfq::cuda::qwen35
