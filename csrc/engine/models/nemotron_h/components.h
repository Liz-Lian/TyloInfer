#pragma once

#include "engine/models/common/mtp.h"
#include "engine/models/nemotron_h/radio_vision.h"
#include "engine/state/media.h"

#include <memory>
#include <optional>
#include <vector>

namespace tyloi::weights { struct Loader; }

namespace tyloi::cuda::nemotron_h {

struct Components {
    std::optional<CudaRadioPromptComponent> radio_vision;
    std::unique_ptr<tyloi::models::Mtp> mtp;

};

Components load_components(Model& model, const tyloi::weights::Loader& weights,
    bool load_optional_components
    );

} // namespace tyloi::cuda::nemotron_h
