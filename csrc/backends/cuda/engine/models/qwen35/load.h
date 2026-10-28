#pragma once

#include "models/qwen35/model.h"
#include "models/qwen35/predictor.h"
#include "mfq_cuda_kv_quantization.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>

struct CudaExecutionContext;
struct TransformerWeightLoader;
namespace mfq::cuda::weight_loader { struct Loader; }

namespace mfq::cuda::qwen35 {

Model load(CudaExecutionContext&, const std::string& model_path,
           const std::string& config_path, std::int64_t context_size = 0,
           bool load_layers = true, bool defer_cache = false,
           std::shared_ptr<const mfq::ModelSource> source = {},
           bool load_input_output = true);

Layer load_layer(const weight_loader::Loader&, const Config&,
                 int index, const std::string& type, bool mirror_scalars = true,
                 std::string_view tensor_root = "model");

std::optional<Predictor> load_predictor(
    TransformerWeightLoader&, const Config&, KvQuantization);

} // namespace mfq::cuda::qwen35
