#pragma once

#include "engine/models/common/mtp.h"
#include "engine/models/nemotron_h/radio_vision.h"

#include <memory>
#include <optional>
#include <vector>

#ifndef TYLOINFER_NATIVE_CUDA
struct CudaExecutionContext;
#endif
namespace tyloi::weights { struct Loader; }

namespace tyloi::cuda::nemotron_h {

struct Components {
    std::optional<CudaRadioPromptComponent> radio_vision;
    std::unique_ptr<tyloi::models::Mtp> mtp;

    tyloi::StepSequence<tyloi::models::Prepared> prepare(Model& language, MediaCache& images,
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
    bool load_optional_components
#ifdef TYLOINFER_NATIVE_CUDA
    );
#else
    , CudaExecutionContext& execution);
#endif

} // namespace tyloi::cuda::nemotron_h
