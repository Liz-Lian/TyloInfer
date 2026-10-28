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

struct LoadOptions {
    std::string config_path;
    std::int64_t context_size = 0;
    int gpu_layers = -1;
    bool extend_yarn = false;
    bool expert_parallel = false;
    bool mirror_scalars = true;
    bool load_layers = true;
    bool load_input_output = true;
};

Model load(std::shared_ptr<const mfq::ModelSource> source,
           const weight_loader::Loader& weights, LayerPlacementConfig& placement,
           mfq::models::YarnScaling& yarn, int& cpu_layers,
           const LoadOptions& options);

Layer load_layer(const weight_loader::Loader&, const Config&,
                 int index, const std::string& type, bool mirror_scalars = true,
                 std::string_view tensor_root = "model");

std::optional<Predictor> load_predictor(
    TransformerWeightLoader&, const Config&, KvQuantization);

} // namespace mfq::cuda::qwen35
