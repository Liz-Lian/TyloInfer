#include "engine/models/groups.h"
#include "engine/models/load.h"
#include "engine/models/source/model_graph.h"
#include "engine/models/deepseek_v4/groups.h"

#include <stdexcept>
#include <utility>

namespace tyloi::models {

std::unique_ptr<Groups> load_groups(std::shared_ptr<const ModelSource> source,
        std::string_view definition, const weights::Loader& weights,
        LayerPlacementConfig& placement, const LoadOptions& options) {
    if (!source || model_plan(graph(*source)).backbone != Backbone::deepseek_v4)
        throw std::runtime_error("streamed KL requires DeepSeek V4");
    return cuda::deepseek_v4::load_groups(std::move(source), definition, weights, placement, options);
}

} // namespace tyloi::models
