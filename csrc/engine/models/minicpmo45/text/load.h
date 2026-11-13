#pragma once

#include "engine/models/minicpmo45/text/model.h"
#include "engine/models/load.h"
#include <string_view>

struct LayerPlacementConfig;
namespace tyloi::weights { struct Loader; }
namespace tyloi::cuda::minicpmo45::text {
/** Load text weights using a resolved source and placement. */
void load(std::unique_ptr<tyloi::models::CausalModel>& owner, std::shared_ptr<const tyloi::ModelSource>, std::string_view definition,
    const tyloi::weights::Loader&, LayerPlacementConfig&, const tyloi::models::LoadOptions&);
Model read(std::shared_ptr<const tyloi::ModelSource>, std::string_view definition, int64_t context);
void load(Model&, const tyloi::weights::Loader&, LayerPlacementConfig&, int gpu_layers = -1,
          bool expert_parallel = false, bool load_layers = true, bool load_input_output = true,
          const tyloi::ops::YarnScaling& yarn = {});
/** Populate an already owned text layer using the selected placement. */
void load_layer(Layer&, const tyloi::weights::Loader&, const Config&, int index, const std::string& type);
} // namespace tyloi::cuda::minicpmo45::text
