#pragma once

#include "engine/models/deepseek_v4/model.h"
#include <string_view>
#include <unordered_set>

struct LayerPlacementConfig;
namespace tyloi::cuda { class Collectives; }
namespace tyloi::weights { struct Loader; }
namespace tyloi::models { struct LoadOptions; }
namespace tyloi::cuda::deepseek_v4 {
/** Load a layer's routed FFN into an owner that retains partially loaded weights. */
void load_ffn(tyloi::cuda::Ffn&, const tyloi::weights::Loader&, const Config&, int, bool);
#ifdef TYLOINFER_NATIVE_CUDA
/** Assemble a supported DeepSeek V4 causal model using the active execution scope. */
void load(std::unique_ptr<tyloi::models::CausalModel>& owner,
    std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
    const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
    const tyloi::models::LoadOptions& options);
#endif
Model read(std::shared_ptr<const tyloi::ModelSource>, std::string_view definition, int64_t context);
void validate_offload(const Config&, const std::unordered_set<int>&);
void load_layer(Layer&, const tyloi::weights::Loader&, const Config&, int index, const std::string& type,
                 Collectives&, int primary, const std::unordered_set<int>&, int64_t& host_bytes);
void load(Model&, const tyloi::weights::Loader&, LayerPlacementConfig&, Collectives&, int primary,
          const std::unordered_set<int>&, int64_t& host_bytes, int gpu_layers);
/** Load shared weights and prepare placement without retaining transformer layers. */
void load_deferred(Model&, const tyloi::weights::Loader&, LayerPlacementConfig&, int gpu_layers);
} // namespace tyloi::cuda::deepseek_v4
