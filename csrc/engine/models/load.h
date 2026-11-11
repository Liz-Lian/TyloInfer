#pragma once

#include "engine/ops.h"

#include <cstdint>
#include <memory>
#include <string_view>
#include <unordered_set>

struct LayerPlacementConfig;

namespace tyloi {
class ModelSource;
namespace weights { struct Loader; }

namespace models {
class CausalModel;

/** Model construction policy without execution resource ownership. */
struct LoadOptions {
    std::int64_t context = 0;
    int gpu_layers = -1;
    bool expert_parallel = false;
    ops::YarnScaling yarn;
    std::unordered_set<int> cpu_offload_layers;
    /** Rows retained by the model's Engram host cache. */
    std::size_t engram_cache_rows = 16'384;
    /** Keep host token IDs for the configured weight prefetch path. */
    bool host_tokens = false;
};

/** Report whether this backbone is assembled through the public causal model factory. */
bool supports(std::string_view backbone);

/** Load a supported causal model from its source, weights and placement. */
std::unique_ptr<CausalModel> load(
    std::shared_ptr<const ModelSource> source,
    std::string_view definition,
    const weights::Loader& weights,
    LayerPlacementConfig& placement,
    const LoadOptions& options = {});

} // namespace models
} // namespace tyloi
