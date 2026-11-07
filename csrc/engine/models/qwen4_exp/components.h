#pragma once

#include "engine/metrics.h"
#include "engine/models/common/mtp.h"
#include "engine/models/common/vision/grid_encoder.h"

#include <memory>
#include <optional>

namespace tyloi::weights { struct Loader; }

namespace tyloi::cuda::qwen4_exp {

struct Model;

struct Components {
    std::optional<grid_vision_runtime::CudaGridVisionPromptComponent> grid_vision;
    std::unique_ptr<tyloi::models::Mtp> mtp;

    static constexpr bool streamed_kv() noexcept { return true; }
    tyloi::engine::Metrics profile_metrics(const Model& language) const;
    std::optional<tyloi::engine::Metrics> memory_metrics(const Model& language) const;
    tyloi::StepSequence<PreparedPrompt> prepare(
        Model& language, MediaCache& images,
#ifndef TYLOINFER_NATIVE_CUDA
        Projector projector, Collectives& collectives,
#endif
        int primary,
#ifndef TYLOINFER_NATIVE_CUDA
        Executor* executor,
#endif
        const std::vector<int64_t>& tokens, const MfqMultimodalInput& media);
};

Components load_components(Model& model, const tyloi::weights::Loader& weights,
    bool load_optional_components);

} // namespace tyloi::cuda::qwen4_exp
