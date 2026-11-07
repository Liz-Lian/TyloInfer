#pragma once

#include "engine/models/qwen35/config.h"
#include "engine/models/common/causal_metadata.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/backends/cuda/executor/task.h"
#include "engine/models/common/model_graph.h"
#include "engine/models/common/capabilities.h"
#include "engine/models/common/attention.h"
#include "engine/backends/cuda/ops/attention/linear.h"
#include "engine/backends/cuda/ops/projection/quant_linear.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/models/source.h"

#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tyloi::cuda::qwen35 {

using DenseTensor = ::tyloi::DenseTensor;
using Config = tyloi::models::qwen35::Config;

struct Full {
    FullAttention attention;
    DenseTensor ffn_norm;
    Ffn ffn;
};

struct Linear {
    LinearAttention attention;
    DenseTensor ffn_norm;
    Ffn ffn;

    Task replay(std::int64_t tokens) const;
};

struct Layer {
    std::variant<Full, Linear> weights;
    int device = 0;
    bool cpu = false;

    std::array<tyloi::cuda::Task, 2> tasks() const;
};

struct Model {
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    CudaModelPlan plan;
    Config config;
    tyloi::models::CausalLmMetadata metadata;
    RopeCache rope;
    RopeCache cpu_rope;
    std::unordered_map<int, RopeCache> device_ropes;
    std::shared_ptr<const ::tyloi::Tensor> embed;
    std::shared_ptr<const ::tyloi::Tensor> lm_head;
    DenseTensor output_norm;
    std::vector<Layer> layers;
#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<tyloi::Tensor> embedding(const tyloi::Tensor& ids, int primary = 0) const;
#endif
};

} // namespace tyloi::cuda::qwen35
