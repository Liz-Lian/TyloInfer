#pragma once
#include "engine/models/glm_dsa/model.h"
#include <string_view>

struct LayerPlacementConfig;
namespace tyloi::weights { struct Loader; }
namespace tyloi::models { struct LoadOptions; }
namespace tyloi::cuda::glm_dsa {
/** Assemble a GLM-DSA causal model with the selected placement and loading policy. */
void load(std::unique_ptr<tyloi::models::CausalModel>& owner,
    std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
    const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
    const tyloi::models::LoadOptions& options);
Model read(std::shared_ptr<const tyloi::ModelSource>, std::string_view definition, int64_t context);
void load(Model&, const ::tyloi::weights::Loader&, LayerPlacementConfig&, int gpu_layers = -1,
          bool load_layers = true, bool load_input_output = true,
          const tyloi::ops::YarnScaling& yarn = {});
/** Populate an already owned layer using the selected placement. */
void load_layer(Layer&, const ::tyloi::weights::Loader&, const Config&, int index, const std::string& type);
} // namespace tyloi::cuda::glm_dsa
