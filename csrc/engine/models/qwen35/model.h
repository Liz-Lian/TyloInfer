#pragma once

#include "config.h"
#include "../common/causal_metadata.h"
#include "backends/cuda/engine/models/ffn.h"
#include "engine/task.h"
#include "formats/model_graph.h"
#include "cuda_model_plan.h"
#include "ops/attention/full.h"
#include "ops/attention/linear.h"
#include "ops/common/include/quant_linear.h"
#include "ops/common/include/rope.h"
#include "source/model_source.h"

#include <cstdint>
#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

struct CudaExecutionContext;

namespace mfq::cuda::qwen35 {

using Tensor = mfq_tensor_backend::Tensor;
using Config = mfq::models::qwen35::Config;

struct Full {
    FullAttention attention;
    Tensor ffn_norm;
    Ffn ffn;
};

struct Linear {
    LinearAttention attention;
    Tensor ffn_norm;
    Ffn ffn;
};

struct Layer {
    std::variant<Full, Linear> weights;
    int device = 0;
    bool cpu = false;

    std::array<mfq::cuda::Task, 2> tasks() const;
};

struct Model {
    std::shared_ptr<const mfq::ModelSource> source;
    mfq::ModelGraph graph;
    CudaModelPlan plan;
    Config config;
    mfq::models::CausalLmMetadata metadata;
    RopeCache rope;
    RopeCache cpu_rope;
    std::unordered_map<int, RopeCache> device_ropes;
    QuantLinear embed;
    QuantLinear lm_head;
    Tensor output_norm;
    std::vector<Layer> layers;
};

Model load(CudaExecutionContext&, const std::string& model_path,
           const std::string& config_path, std::int64_t context_size = 0,
           bool load_layers = true, bool defer_cache = false,
           std::shared_ptr<const mfq::ModelSource> source = {},
           bool load_input_output = true);

Layer load_layer(CudaExecutionContext&, const mfq::ModelSource&, const Config&,
                 int index, const std::string& type, int device, bool cpu,
                 std::string_view tensor_root = "model");

} // namespace mfq::cuda::qwen35
