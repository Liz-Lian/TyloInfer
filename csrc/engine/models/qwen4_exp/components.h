#pragma once

#include "engine/metrics.h"
#include "engine/ops.h"
#include "engine/models/common/mtp.h"
#include "engine/models/common/vision/grid_encoder.h"
#include "engine/state/media.h"

#include <memory>
#include <optional>
#include <string_view>

namespace tyloi::weights { struct Loader; }

namespace tyloi::cuda::qwen4_exp {

struct Model;
class State;

struct Components {
    std::optional<grid_vision_runtime::CudaGridVisionPromptComponent> grid_vision;
    std::unique_ptr<tyloi::models::Mtp> mtp;

    static constexpr bool streamed_kv() noexcept { return true; }
#ifndef TYLOINFER_NATIVE_CUDA
    /** Return graph profiling metrics for the request state. */
    tyloi::engine::Metrics profile_metrics(const Model& language, const State& state) const;
    std::optional<tyloi::engine::Metrics> memory_metrics(const Model& language, const State& state) const;
#endif
    /** Prepare supported media using the request's cache and model weights. */
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
};

/** Load supported optional components using the model's configuration. */
Components load_components(Model& model, const tyloi::weights::Loader& weights,
    bool load_optional_components
#ifdef TYLOINFER_NATIVE_CUDA
    , std::string_view definition, const tyloi::ops::YarnScaling& yarn
#endif
    );

} // namespace tyloi::cuda::qwen4_exp
