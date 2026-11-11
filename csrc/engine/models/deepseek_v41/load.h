#pragma once

#include <memory>
#include <string_view>

struct LayerPlacementConfig;
namespace tyloi { class ModelSource; }
namespace tyloi::models { class CausalModel; struct LoadOptions; }
namespace tyloi::weights { struct Loader; }

namespace tyloi::cuda::deepseek_v41 {
#ifdef TYLOINFER_NATIVE_CUDA
/** Assemble a DeepSeek V4.1 model from its source and placed weights. */
std::unique_ptr<tyloi::models::CausalModel> load(
    std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
    const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
    const tyloi::models::LoadOptions& options);
#endif
} // namespace tyloi::cuda::deepseek_v41
