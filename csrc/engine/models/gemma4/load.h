#pragma once

#include "engine/models/gemma4/model.h"
#include <string>

struct LayerPlacementConfig;
namespace mfq::cuda::weight_loader { struct Loader; }

namespace mfq::cuda::gemma4 {

void load(Model&, const weight_loader::Loader&, LayerPlacementConfig&, int gpu_layers = -1,
          bool expert_parallel = false, bool load_layers = true, bool load_input_output = true);
Layer load_layer(const weight_loader::Loader&, const Config&, int index, const std::string& type);

} // namespace mfq::cuda::gemma4
