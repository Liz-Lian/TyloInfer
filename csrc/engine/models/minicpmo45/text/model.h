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

namespace mfq::cuda::minicpmo45::text {
using Config = mfq::models::minicpmo45::Config;

struct Model {
    std::shared_ptr<const mfq::ModelSource> source;
    mfq::ModelGraph graph;
    CudaModelPlan plan;
    Config config;
    mfq::models::CausalLmMetadata metadata;
    QuantLinear embed, lm_head;
    Tensor output_norm;
    RopeCache rope, cpu_rope;
    std::unordered_map<int, RopeCache> device_ropes;
    std::vector<Layer> layers;
};
} // namespace mfq::cuda::minicpmo45::text
