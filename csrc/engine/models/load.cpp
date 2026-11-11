#include "engine/models/load.h"

#include "engine/models/causal_model.h"
#include "engine/models/gemma4/load.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/models/glm_dsa/load.h"
#include "engine/models/deepseek_v4/load.h"
#include "engine/models/glm5_next/load.h"
#include "engine/models/nemotron_h/load.h"
#endif
#include "engine/models/source.h"
#include "engine/models/source/model_graph.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace tyloi::models {
namespace {

enum class Kind {
    gemma4,
#ifdef TYLOINFER_NATIVE_CUDA
    glm_dsa, deepseek_v4, glm5_next, nemotron_h,
#endif
    unsupported
};

Kind select(std::string_view backbone) {
    if (backbone == "gemma4") return Kind::gemma4;
#ifdef TYLOINFER_NATIVE_CUDA
    if (backbone == "glm_dsa") return Kind::glm_dsa;
    if (backbone == "deepseek_v4") return Kind::deepseek_v4;
    if (backbone == "glm5_next") return Kind::glm5_next;
    if (backbone == "nemotron_h") return Kind::nemotron_h;
#endif
    return Kind::unsupported;
}

} // namespace

bool supports(std::string_view backbone) { return select(backbone) != Kind::unsupported; }

std::unique_ptr<CausalModel> load(
    std::shared_ptr<const ModelSource> source,
    std::string_view definition,
    const weights::Loader& weights,
    LayerPlacementConfig& placement,
    const LoadOptions& options) {
    const auto model_graph = graph(*source);
    switch (select(model_graph.backbone)) {
    case Kind::gemma4:
        if (!options.cpu_offload_layers.empty())
            throw std::invalid_argument("Gemma4 does not support --cpu-offload-layers");
        return ::tyloi::cuda::gemma4::load(
            std::move(source), definition, options.context, weights, placement,
            options.gpu_layers, options.expert_parallel);
#ifdef TYLOINFER_NATIVE_CUDA
    case Kind::glm_dsa:
        return ::tyloi::cuda::glm_dsa::load(std::move(source), definition, weights, placement, options);
    case Kind::deepseek_v4:
        return ::tyloi::cuda::deepseek_v4::load(std::move(source), definition, weights, placement, options);
    case Kind::glm5_next:
        return ::tyloi::cuda::glm5_next::load(std::move(source), definition, weights, placement, options);
    case Kind::nemotron_h:
        return ::tyloi::cuda::nemotron_h::load(std::move(source), definition, weights, placement, options);
#endif
    case Kind::unsupported:
        break;
    }
    throw std::runtime_error(
        "unsupported model backbone: " + model_graph.backbone);
}

} // namespace tyloi::models
