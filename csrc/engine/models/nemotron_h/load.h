#pragma once

#include "engine/models/nemotron_h/model.h"

#include <cstdint>
#include <string>
#include <string_view>

struct LayerPlacementConfig;
namespace tyloi::weights { struct Loader; }

namespace tyloi::cuda::nemotron_h {

Model read(std::shared_ptr<const tyloi::ModelSource>, std::string_view definition, std::int64_t context);
void load(Model&, const ::tyloi::weights::Loader&, LayerPlacementConfig&,
          int gpu_layers = -1, bool expert_parallel = false, bool load_layers = true,
          bool load_input_output = true);
Layer load_layer(const ::tyloi::weights::Loader&, const Config&, int index,
                 const std::string& type, std::string_view tensor_root = "model");

} // namespace tyloi::cuda::nemotron_h
