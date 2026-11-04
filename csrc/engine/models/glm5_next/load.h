#pragma once

#include "engine/models/glm5_next/model.h"
#include <string_view>

struct LayerPlacementConfig;
namespace tyloi::cuda::weight_loader { struct Loader; }

namespace tyloi::cuda::glm5_next {
Model read(std::shared_ptr<const tyloi::ModelSource>, std::string_view definition, int64_t context);
void load(Model&, const weight_loader::Loader&, LayerPlacementConfig&,
          int gpu_layers = -1, bool load_layers = true,
          bool load_input_output = true);
Layer load_layer(const weight_loader::Loader&, const tyloi::models::glm5_next::Config&, int index);
} // namespace tyloi::cuda::glm5_next
