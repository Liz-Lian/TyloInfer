#pragma once

#include "models/common/grid_vision.h"
#include "models/qwen35/model.h"
#include "models/qwen35/mtp.h"
#include "native/cuda/profile.h"

#include <cstdint>
#include <memory>
#include <optional>

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

    CudaComponentState state() const noexcept {
        return cuda_component_state(
            graph, plan, grid_vision.has_value(), bool(mtp));
    }
};

Components load_components(
    const Model& model,
    const weight_loader::Loader& weights,
    const Executor& executor,
    Collectives& copies,
    const int64_t& kv_capacity,
    Trace* const& trace,
    const int& trace_scope,
    bool optional,
    const LayerPlacementConfig& placement,
    int cpu_layers,
    bool cpu_offload,
    bool preallocate);

} // namespace mfq::cuda::qwen35
