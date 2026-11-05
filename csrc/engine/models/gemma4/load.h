#pragma once

#include "engine/models/gemma4/model.h"
#include <string>

struct LayerPlacementConfig;
namespace tyloi::weights { struct Loader; }

namespace tyloi::cuda::gemma4 {

void load(Model&, const tyloi::weights::Loader&, LayerPlacementConfig&, int gpu_layers = -1,
          bool expert_parallel = false, bool load_layers = true, bool load_input_output = true);
Layer load_layer(const tyloi::weights::Loader&, const Config&, int index, const std::string& type);

} // namespace tyloi::cuda::gemma4
