#pragma once

#include "engine/models/gemma4/config.h"
#include <memory>
#include <string>
#include <string_view>

struct LayerPlacementConfig;
namespace tyloi::weights { struct Loader; }
namespace tyloi { class ModelSource; }
namespace tyloi::models { class CausalModel; }

namespace tyloi::cuda::gemma4 {

struct Model;
struct Layer;

std::unique_ptr<tyloi::models::CausalModel> load(
    std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
    std::int64_t context, const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
    int gpu_layers = -1, bool expert_parallel = false);

void load(Model&, const tyloi::weights::Loader&, LayerPlacementConfig&, int gpu_layers = -1,
          bool expert_parallel = false, bool load_layers = true, bool load_input_output = true);
Layer load_layer(const tyloi::weights::Loader&, const tyloi::models::gemma4::Config&,
                 int index, const std::string& type);

} // namespace tyloi::cuda::gemma4
