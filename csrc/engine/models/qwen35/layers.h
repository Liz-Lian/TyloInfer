#pragma once

#include "engine/models/layers.h"

namespace tyloi::cuda::qwen35 {

/** Assemble Qwen layer-wise computation with bounded input and output weight reads. */
std::unique_ptr<tyloi::models::Layers> load_layers(
    std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
    const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
    const tyloi::models::LoadOptions& options);

} // namespace tyloi::cuda::qwen35
