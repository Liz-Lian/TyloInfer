#pragma once

#include "models/qwen35/model.h"
#include "models/qwen35/predictor.h"
#include "engine/models/common/yarn.h"
#include "mfq_cuda_kv_quantization.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

struct LayerPlacementConfig;
struct TransformerWeightLoader;
namespace mfq::cuda::weight_loader { struct Loader; }

namespace mfq::cuda::qwen35 {

Model read(std::shared_ptr<const mfq::ModelSource> source,
           const std::string& config_path, std::int64_t context,
           mfq::models::YarnScaling& yarn, bool extend_yarn);

int place(const Model&, const weight_loader::Loader&, LayerPlacementConfig&,
          int gpu_layers, bool expert_parallel);

void load(Model&, const weight_loader::Loader&, const LayerPlacementConfig&,
          int cpu_layers, const mfq::models::YarnScaling&, bool load_layers,
          bool load_input_output, bool mirror_scalars = true);

Layer load_layer(const weight_loader::Loader&, const Config&,
                 int index, const std::string& type, bool mirror_scalars = true,
                 std::string_view tensor_root = "model");

std::optional<Predictor> load_predictor(
    TransformerWeightLoader&, const Config&, KvQuantization);

} // namespace mfq::cuda::qwen35
