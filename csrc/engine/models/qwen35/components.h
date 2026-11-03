#pragma once
#include "engine/models/source.h"
#include "engine/ops.h"

#include "engine/engine.h"
#include "engine/models/common/vision/grid_encoder.h"
#include "engine/models/qwen35/model.h"
#include "engine/models/qwen35/mtp.h"
#include "engine/backends/cuda/native/profile.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace mfq::cuda::weight_loader {
struct Loader;
}

namespace mfq::cuda {
class Collectives;
class Executor;
}

struct LayerPlacementConfig;

namespace mfq::cuda::qwen35 {

struct Components {
    mfq::ModelGraph graph;
    CudaModelPlan plan;
    std::optional<grid_vision_runtime::CudaGridVisionPromptComponent>
        grid_vision;
    std::unique_ptr<Mtp> mtp;

    static Components load(Model& model, const weight_loader::Loader& weights,
        LayerPlacementConfig& placement, int& cpu_layers, Executor& executor,
        Collectives& copies, const int64_t& kv_capacity, Trace* const& trace,
        const int& trace_scope, bool optional, bool cpu_offload, bool preallocate);

    CudaComponentState state() const noexcept {
        return cuda_component_state(
            graph, plan, grid_vision.has_value(), bool(mtp));
    }

    static constexpr bool streamed_kv() noexcept { return false; }

    mfq::engine::Metrics metrics() const;

    void describe(mfq::engine::EngineInfo& info) const;

    mfq::StepSequence<PreparedPrompt> prepare(
        Model& language, Projector projector, std::shared_ptr<MoeExpertCache> cache,
        Collectives& collectives, int primary, Executor* executor,
        const std::vector<int64_t>& tokens, const MfqMultimodalInput& media);

    template <class T>
    mfq::engine::ControlResult control(const T&) {
        throw std::invalid_argument("model has no duplex component");
    }
};

} // namespace mfq::cuda::qwen35
