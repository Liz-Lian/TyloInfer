#pragma once

#include "engine/models/minicpmo45/layer.h"
#include "engine/models/minicpmo45/config.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/models/common/causal_metadata.h"
#include "engine/models/common/capabilities.h"
#include "engine/models/source.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace tyloi::cuda::minicpmo45::text {
using Config = tyloi::models::minicpmo45::Config;

struct Model {
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    CudaModelPlan plan;
    Config config;
    tyloi::models::CausalLmMetadata metadata;
    QuantLinear embed, lm_head;
    Tensor output_norm;
    RopeCache rope, cpu_rope;
    std::unordered_map<int, RopeCache> device_ropes;
    std::vector<Layer> layers;
};
} // namespace tyloi::cuda::minicpmo45::text
