#pragma once

#include "engine/models/qwen35/config.h"
#include "engine/models/metadata.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/models/source/model_graph.h"
#include "engine/models/common/capabilities.h"
#include "engine/models/common/attention.h"
#include "engine/backends/cuda/ops/attention/linear.h"
#include "engine/backends/cuda/ops/projection/quant_linear.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/models/source.h"

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
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor ffn_norm;
#else
    DenseTensor ffn_norm;
#endif
    Ffn ffn;
};

struct Linear {
    LinearAttention attention;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor ffn_norm;
#else
    DenseTensor ffn_norm;
#endif
    Ffn ffn;
};

struct Layer {
    std::variant<Full, Linear> weights;
    int device = 0;
    bool cpu = false;
};

struct Model {
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    CudaModelPlan plan;
    Config config;
    tyloi::models::CausalLmMetadata metadata;
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::ops::RopeTable rope, cpu_rope;
    std::unordered_map<int, tyloi::ops::RopeTable> device_ropes;
#else
    RopeCache rope;
    RopeCache cpu_rope;
    std::unordered_map<int, RopeCache> device_ropes;
#endif
    ::tyloi::Tensor embed;
    ::tyloi::Tensor lm_head;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor output_norm;
#else
    DenseTensor output_norm;
#endif
    std::vector<Layer> layers;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor embedding(const tyloi::Tensor& ids, int primary = 0) const;
#endif
};

} // namespace tyloi::cuda::qwen35
