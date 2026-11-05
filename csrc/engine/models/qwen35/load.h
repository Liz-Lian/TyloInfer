#pragma once
#include "engine/models/source.h"

#include "engine/models/qwen35/model.h"
#include "engine/models/qwen35/predictor.h"
#include "engine/ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_kv_quantization.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

struct LayerPlacementConfig;
struct TransformerWeightLoader;
namespace tyloi::weights { struct Loader; }

namespace tyloi::cuda::qwen35 {

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

Model load(std::shared_ptr<const tyloi::ModelSource> source,
           const ::tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
           tyloi::ops::YarnScaling& yarn, int& cpu_layers,
           const LoadOptions& options);

Layer load_layer(const ::tyloi::weights::Loader&, const Config&,
                 int index, const std::string& type, bool mirror_scalars = true,
                 std::string_view tensor_root = "model");

std::optional<Predictor> load_predictor(
    TransformerWeightLoader&, const Config&, KvQuantization);

} // namespace tyloi::cuda::qwen35
