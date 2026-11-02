#pragma once

#include "engine/models/minicpmo45/text/model.h"
#include <string_view>

struct LayerPlacementConfig;
namespace mfq::cuda::weight_loader { struct Loader; }
namespace mfq::cuda::minicpmo45::text {
Model read(std::shared_ptr<const mfq::ModelSource>, std::string_view definition, int64_t context);
void load(Model&, const weight_loader::Loader&, LayerPlacementConfig&, int gpu_layers = -1,
          bool expert_parallel = false, bool load_layers = true, bool load_input_output = true,
          const mfq::ops::YarnScaling& yarn = {});
Layer load_layer(const weight_loader::Loader&, const Config&, int index, const std::string& type);
} // namespace mfq::cuda::minicpmo45::text
