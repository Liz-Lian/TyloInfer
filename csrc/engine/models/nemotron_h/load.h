#pragma once

#include "engine/models/nemotron_h/model.h"

#include <cstdint>
#include <string>
#include <string_view>

struct LayerPlacementConfig;
namespace tyloi::weights { struct Loader; }
namespace tyloi::models { struct LoadOptions; }

namespace tyloi::cuda::nemotron_h {

/** Load native model weights and optional components through the public factory. */
void load(std::unique_ptr<tyloi::models::CausalModel>& owner, std::shared_ptr<const tyloi::ModelSource>,
    std::string_view definition, const tyloi::weights::Loader&, LayerPlacementConfig&,
    const tyloi::models::LoadOptions&);
Model read(std::shared_ptr<const tyloi::ModelSource>, std::string_view definition, std::int64_t context);
void load(Model&, const ::tyloi::weights::Loader&, LayerPlacementConfig&,
          int gpu_layers = -1, bool expert_parallel = false, bool load_layers = true,
          bool load_input_output = true);
/** Populate an already owned layer using the selected placement. */
void load_layer(Layer&, const ::tyloi::weights::Loader&, const Config&, int index,
                const std::string& type, std::string_view tensor_root = "model");
/** Construct a predictor layer through the existing component loader. */
Layer load_layer(const ::tyloi::weights::Loader&, const Config&, int index,
                 const std::string& type, std::string_view tensor_root = "model");

} // namespace tyloi::cuda::nemotron_h
