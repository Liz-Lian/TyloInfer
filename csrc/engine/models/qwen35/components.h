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
#ifndef TYLOINFER_NATIVE_CUDA
class Cache;
#endif
struct Loader;
}

#ifndef TYLOINFER_NATIVE_CUDA
namespace tyloi::cuda {
class Collectives;
class Executor;
}
#endif

struct LayerPlacementConfig;

namespace tyloi::cuda::qwen35 {

struct Components {
    tyloi::ModelGraph graph;
    ::tyloi::models::Plan plan;
    std::optional<grid_vision_runtime::CudaGridVisionPromptComponent>
        grid_vision;
    std::unique_ptr<Mtp> mtp;

#ifdef TYLOINFER_NATIVE_CUDA
    /** Load available vision and predictor weights using the resolved placement policy. */
    static Components load(Model& model, const ::tyloi::weights::Loader& weights,
        LayerPlacementConfig& placement, int cpu_layers, bool optional, bool cpu_offload,
        bool transfer_auto);
#else
    /** Load available vision and predictor components using the legacy execution owner. */
    static Components load(Model& model, const ::tyloi::weights::Loader& weights, ::tyloi::weights::Cache& cache,
        LayerPlacementConfig& placement, int& cpu_layers, Executor& executor,
        Collectives& copies, const int64_t& kv_capacity, Trace* const& trace,
        const int& trace_scope, bool optional, bool cpu_offload, bool preallocate);
#endif

    ::tyloi::models::ComponentState state() const noexcept {
        return ::tyloi::models::component_state(
            graph, plan, grid_vision.has_value(), bool(mtp));
    }

    static constexpr bool streamed_kv() noexcept { return false; }

    tyloi::engine::Metrics metrics() const;

    void describe(tyloi::engine::EngineInfo& info) const;

    /** Prepare media using the request's image cache and the assembled vision weights. */
    tyloi::StepSequence<tyloi::models::Prepared> prepare(
#ifdef TYLOINFER_NATIVE_CUDA
        const Model& language, tyloi::cache::Media& images,
#else
        Model& language, tyloi::cache::Media& images,
#endif
#ifndef TYLOINFER_NATIVE_CUDA
        Projector projector, Collectives& collectives,
#endif
        int primary,
#ifndef TYLOINFER_NATIVE_CUDA
        Executor* executor,
#endif
        const std::vector<int64_t>& tokens, const MfqMultimodalInput& media)
#ifdef TYLOINFER_NATIVE_CUDA
        const
#endif
        ;

    template <class T>
    tyloi::engine::ControlResult control(const T&) {
        throw std::invalid_argument("model has no duplex component");
    }
};

} // namespace tyloi::cuda::qwen35
