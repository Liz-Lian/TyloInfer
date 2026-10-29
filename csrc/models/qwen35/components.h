#pragma once

#include "engine/executor/loaded.h"
#include "engine/executor/engine.h"
#include "models/common/grid_vision.h"
#include "models/qwen35/model.h"
#include "models/qwen35/mtp.h"
#include "native/cuda/profile.h"

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

namespace mfq::models {
struct YarnScaling;
}

struct LayerPlacementConfig;

namespace mfq::cuda::qwen35 {

struct LoadOptions;
class State;

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
    bool preallocate);

} // namespace mfq::cuda::qwen35
