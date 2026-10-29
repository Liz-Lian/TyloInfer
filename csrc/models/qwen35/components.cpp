#include "models/qwen35/components.h"

#include "backends/cuda/engine/models/load.h"
#include "models/common/grid_vision.h"
#include "models/qwen35/forward.h"
#include "models/qwen35/load.h"
#include "models/qwen35/mtp.h"
#include "models/qwen35/state.h"

#include "native/cuda/collectives.h"
#include "engine/executor/run.h"
#include "backends/cuda/engine/weights/placement.h"

#include <iostream>
#include <stdexcept>
#include <utility>

namespace mfq::cuda::qwen35 {

std::optional<KvQuantization> kv_quantization(const Model&) {
    return cuda_kv_quantization();
}

Loaded<Model, State, Components> assemble(
    std::shared_ptr<const mfq::ModelSource> source,
    const weight_loader::Loader& weights,
    LayerPlacementConfig& placement,
    mfq::models::YarnScaling& yarn,
    int& cpu_layers,
    const LoadOptions& options,
    std::unique_ptr<Executor> executor,
    Collectives& copies,
    const bool& transposed,
    const int64_t& kv_capacity,
    Trace* const& trace,
    const int& trace_scope,
    bool optional,
    bool cpu_offload,
    bool preallocate) {
    Loaded<Model, State, Components> result{
        std::make_shared<Model>(load(
            source, weights, placement, yarn, cpu_layers, options)), {}, {},
        std::move(executor)};
    auto& model = *result.model;
    result.owned_state = std::make_unique<State>(model, transposed);
    if (preallocate) result.owned_state->preallocate();
    result.components = Components{model.graph, model.plan};
    if (!optional) return result;
    auto& components = result.components;
    const bool mtp_allowed =
        !placement.enabled() && cpu_layers == 0 && !cpu_offload &&
        !weights.cache.moe;

    if (components.plan.vision == CudaVisionAdapter::grid_vit) {
        const auto* component = components.graph.component("vision");
        const auto& config = model.config;
        if (component == nullptr || !config.grid_vision ||
            !config.image_token_id || !config.video_token_id) {
            throw std::runtime_error(
                "CUDA grid-Vision configuration is incomplete");
        }
        components.grid_vision.emplace(
            grid_vision_runtime::CudaGridVisionPromptComponent::load(
                weights, *config.grid_vision,
                *config.image_token_id, *config.video_token_id,
                component->input_contract, component->position_policy));
    } else if (components.plan.vision != CudaVisionAdapter::none) {
        throw std::runtime_error("CUDA vision adapter is unsupported for Qwen");
    }

    if (components.plan.predictor == CudaPredictorAdapter::qwen35) {
        if (mtp_allowed && model.metadata.num_experts == 0 &&
            supports_speculation(model)) {
            TransformerWeightLoader loader{weights};
            auto predictor = load_predictor(
                loader, model.config, cuda_kv_quantization());
            if (predictor) {
                // Mtp retains these references, so they must outlive the result.
                components.mtp = std::make_unique<Mtp>(
                    std::move(*predictor), model.embed, model.rope,
                    result.executor->fork(), copies, weights.placement.device,
                    kv_capacity, trace, trace_scope, preallocate);
            }
        } else {
            std::cerr
                << "qwen_mtp unavailable: CUDA adapter requires dense GPU-resident Qwen blocks\n";
        }
    } else if (components.plan.predictor != CudaPredictorAdapter::none) {
        throw std::runtime_error("CUDA predictor adapter is unsupported for Qwen");
    }
    return result;
}

} // namespace mfq::cuda::qwen35
