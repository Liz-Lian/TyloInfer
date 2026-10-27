#pragma once

#include "engine/models/qwen35/config.h"
#include "engine/models/common/causal_metadata.h"
#include "backends/cuda/engine/models/ffn.h"
#include "engine/task.h"
#include "formats/model_graph.h"
#include "cuda_model_plan.h"
#include "ops/attention/full.h"
#include "ops/attention/linear.h"
#include "ops/common/include/quant_linear.h"
#include "ops/common/include/rope.h"
#include "source/model_source.h"

#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>

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

    Task replay(std::int64_t tokens) const;
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

} // namespace mfq::cuda::qwen35
