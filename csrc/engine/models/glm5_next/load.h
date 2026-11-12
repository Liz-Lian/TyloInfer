#pragma once

#include "engine/models/glm5_next/model.h"
#include <string_view>

struct LayerPlacementConfig;
namespace tyloi::weights { struct Loader; }
namespace tyloi::models { struct LoadOptions; }

namespace tyloi::cuda::glm5_next {
#ifdef TYLOINFER_NATIVE_CUDA
/** Assemble a GLM5-Next causal model and its supported predictor. */
void load(std::unique_ptr<tyloi::models::CausalModel>& owner,
    std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
    const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
    const tyloi::models::LoadOptions& options);
#endif
Model read(std::shared_ptr<const tyloi::ModelSource>, std::string_view definition, int64_t context);
void load(Model&, const ::tyloi::weights::Loader&, LayerPlacementConfig&,
          int gpu_layers = -1, bool load_layers = true,
          bool load_input_output = true);
/** Populate an already owned layer using the selected placement. */
void load_layer(Layer&, const ::tyloi::weights::Loader&, const tyloi::models::glm5_next::Config&, int index);
} // namespace tyloi::cuda::glm5_next
