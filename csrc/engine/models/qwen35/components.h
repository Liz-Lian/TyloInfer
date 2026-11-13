#pragma once
#include "engine/models/source.h"
#include "engine/ops.h"

#include "engine/engine.h"
#include "engine/models/common/vision/grid_encoder.h"
#include "engine/state/media.h"
#include "engine/models/qwen35/model.h"
#include "engine/models/qwen35/mtp.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace tyloi::weights {
struct Loader;
}


struct LayerPlacementConfig;

namespace tyloi::cuda::qwen35 {

struct Components {
    tyloi::ModelGraph graph;
    ::tyloi::models::Plan plan;
    std::optional<grid_vision_runtime::CudaGridVisionPromptComponent>
        grid_vision;
    std::unique_ptr<Mtp> mtp;

    /** Load available vision and predictor weights using the resolved placement policy. */
    static Components load(Model& model, const ::tyloi::weights::Loader& weights,
        LayerPlacementConfig& placement, int cpu_layers, bool optional, bool cpu_offload,
        bool transfer_auto);

    ::tyloi::models::ComponentState state() const noexcept {
        return ::tyloi::models::component_state(
            graph, plan, grid_vision.has_value(), bool(mtp));
    }

    static constexpr bool streamed_kv() noexcept { return false; }

    tyloi::engine::Metrics metrics() const;

    void describe(tyloi::engine::EngineInfo& info) const;

    /** Prepare media using the request's image cache and the assembled vision weights. */
    tyloi::StepSequence<tyloi::models::Prepared> prepare(
        const Model& language, tyloi::cache::Media& images,
        int primary,
        const std::vector<int64_t>& tokens, const MfqMultimodalInput& media)
        const
        ;

    template <class T>
    tyloi::engine::ControlResult control(const T&) {
        throw std::invalid_argument("model has no duplex component");
    }
};

} // namespace tyloi::cuda::qwen35
