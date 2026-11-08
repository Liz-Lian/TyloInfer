#pragma once

#include "engine/models/gemma4/model.h"
#include <memory>
#include <string>
#include <string_view>

struct LayerPlacementConfig;
namespace tyloi::weights { struct Loader; }

namespace tyloi::cuda::gemma4 {

#ifdef TYLOINFER_NATIVE_CUDA
std::shared_ptr<tyloi::models::CausalModel> load(
    std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
    std::int64_t context, const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
    int gpu_layers = -1, bool expert_parallel = false);
#endif

void load(Model&, const tyloi::weights::Loader&, LayerPlacementConfig&, int gpu_layers = -1,
          bool expert_parallel = false, bool load_layers = true, bool load_input_output = true);
Layer load_layer(const tyloi::weights::Loader&, const Config&, int index, const std::string& type);

} // namespace tyloi::cuda::gemma4
