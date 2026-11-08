#include "engine/models/load.h"

#include "engine/models/causal_model.h"
#include "engine/models/gemma4/load.h"
#include "engine/models/source.h"
#include "engine/models/source/model_graph.h"

#include <stdexcept>
#include <utility>

namespace tyloi::models {

std::unique_ptr<CausalModel> load(
    std::shared_ptr<const ModelSource> source,
    std::string_view definition,
    std::int64_t context,
    const weights::Loader& weights,
    LayerPlacementConfig& placement,
    int gpu_layers,
    bool expert_parallel) {
    const auto model_graph = graph(*source);
    if (model_graph.backbone == "gemma4") {
        return ::tyloi::cuda::gemma4::load(
            std::move(source), definition, context, weights, placement,
            gpu_layers, expert_parallel);
    }
    throw std::runtime_error(
        "unsupported model backbone: " + model_graph.backbone);
}

} // namespace tyloi::models
