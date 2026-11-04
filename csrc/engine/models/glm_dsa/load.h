#pragma once
#include "engine/models/glm_dsa/model.h"
#include <string_view>

struct LayerPlacementConfig;
namespace tyloi::cuda::weight_loader { struct Loader; }
namespace tyloi::cuda::glm_dsa {
Model read(std::shared_ptr<const tyloi::ModelSource>, std::string_view definition, int64_t context);
void load(Model&, const weight_loader::Loader&, LayerPlacementConfig&, int gpu_layers = -1,
          bool load_layers = true, bool load_input_output = true,
          const tyloi::ops::YarnScaling& yarn = {});
Layer load_layer(const weight_loader::Loader&, const Config&, int index, const std::string& type);
} // namespace tyloi::cuda::glm_dsa
