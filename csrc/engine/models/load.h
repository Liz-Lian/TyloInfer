#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

struct LayerPlacementConfig;

namespace tyloi {
class ModelSource;
namespace weights { struct Loader; }

namespace models {
class CausalModel;

/** Load a supported causal model from its source, weights and placement. */
std::unique_ptr<CausalModel> load(
    std::shared_ptr<const ModelSource> source,
    std::string_view definition,
    std::int64_t context,
    const weights::Loader& weights,
    LayerPlacementConfig& placement,
    int gpu_layers = -1,
    bool expert_parallel = false);

} // namespace models
} // namespace tyloi
