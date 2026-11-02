#pragma once

#include "engine/models/deepseek_v4/model.h"
#include <string_view>
#include <unordered_set>

struct LayerPlacementConfig;
namespace mfq::cuda { class Collectives; }
namespace mfq::cuda::weight_loader { struct Loader; }
namespace mfq::cuda::deepseek_v4 {
Model read(std::shared_ptr<const mfq::ModelSource>, std::string_view definition, int64_t context);
void validate_offload(const Config&, const std::unordered_set<int>&);
Layer load_layer(const weight_loader::Loader&, const Config&, int index, const std::string& type,
                 Collectives&, int primary, const std::unordered_set<int>&, int64_t& host_bytes);
void load(Model&, const weight_loader::Loader&, LayerPlacementConfig&, Collectives&, int primary,
          const std::unordered_set<int>&, int64_t& host_bytes, int gpu_layers);
} // namespace mfq::cuda::deepseek_v4
