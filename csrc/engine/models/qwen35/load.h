#pragma once
#include "engine/models/source.h"

#include "engine/models/qwen35/model.h"
#include "engine/models/qwen35/predictor.h"
#include "engine/models/load.h"
#include "engine/ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_kv_quantization.h"
#include "engine/state/kv_view.h"

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

#ifdef TYLOINFER_NATIVE_CUDA
/** Load Qwen weights using an already resolved model definition and YaRN policy. */
std::unique_ptr<Model> load(std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
           const ::tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
           const tyloi::models::LoadOptions& options);
#else
/** Load Qwen weights using the configured model definition. */
Model load(std::shared_ptr<const tyloi::ModelSource> source,
           const ::tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
           tyloi::ops::YarnScaling& yarn, int& cpu_layers,
           const LoadOptions& options);
#endif

Layer load_layer(const ::tyloi::weights::Loader&, const Config&,
                 int index, const std::string& type, bool mirror_scalars = true,
                 std::string_view tensor_root = "model");

std::optional<Predictor> load_predictor(
    TransformerWeightLoader&, const Config&, tyloi::state::KvQuantization);

} // namespace tyloi::cuda::qwen35
