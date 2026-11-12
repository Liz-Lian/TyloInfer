#pragma once

#include <memory>
#include <string_view>

struct LayerPlacementConfig;
namespace tyloi { class ModelSource; }
namespace tyloi::weights { struct Loader; }
namespace tyloi::models { class Layers; struct LoadOptions; }

namespace tyloi::cuda::gemma4 {

/** Assemble the bounded, layer-wise Gemma4 reference model. */
std::unique_ptr<tyloi::models::Layers> load_layers(
    std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
    const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
    const tyloi::models::LoadOptions& options);

} // namespace tyloi::cuda::gemma4
