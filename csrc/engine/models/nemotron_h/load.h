#pragma once

#include "engine/models/nemotron_h/model.h"

#include <cstdint>
#include <string>
#include <string_view>

struct LayerPlacementConfig;
namespace mfq::cuda { class Pipelines; }
namespace mfq::cuda::weight_loader { struct Loader; }

namespace mfq::cuda::nemotron_h {

Model read(std::shared_ptr<const mfq::ModelSource>, std::string_view definition, std::int64_t context);
void load(Model&, const weight_loader::Loader&, Pipelines&, LayerPlacementConfig&,
          int gpu_layers = -1, bool expert_parallel = false, bool load_layers = true,
          bool load_input_output = true);
Layer load_layer(Pipelines&, const weight_loader::Loader&, const Config&, int index,
                 const std::string& type, std::string_view tensor_root = "model");

} // namespace mfq::cuda::nemotron_h
