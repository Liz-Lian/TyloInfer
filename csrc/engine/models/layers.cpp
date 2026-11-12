#include "engine/models/layers.h"
#include "engine/models/load.h"
#include "engine/models/source/model_graph.h"
#include "engine/models/qwen35/layers.h"
#include "engine/models/gemma4/layers.h"

#include <stdexcept>
#include <utility>

namespace tyloi::models {

void load_layers(std::unique_ptr<Layers>& owner, std::shared_ptr<const ModelSource> source,
        std::string_view definition, const weights::Loader& weights,
        LayerPlacementConfig& placement, const LoadOptions& options) {
    if (owner) throw std::invalid_argument("layer owner must be empty");
    if (!source) throw std::invalid_argument("layer-wise loading requires a model source");
    switch (model_plan(graph(*source)).backbone) {
        case Backbone::generic_qwen:
            return cuda::qwen35::load_layers(owner, std::move(source), definition, weights, placement, options);
        case Backbone::gemma4:
            return cuda::gemma4::load_layers(owner, std::move(source), definition, weights, placement, options);
        default:
            throw std::runtime_error("native reference generation supports original Qwen3.5 and Gemma4");
    }
}

} // namespace tyloi::models
