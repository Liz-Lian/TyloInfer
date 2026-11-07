#pragma once

#include "engine/models/glm_dsa/config.h"
#include "engine/models/common/causal_metadata.h"
#include "engine/models/common/capabilities.h"
#include "engine/models/source.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/weights/moe/experts.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace tyloi::cuda::glm_dsa {
using DenseTensor = ::tyloi::DenseTensor;
using Config = tyloi::models::glm_dsa::Config;

struct Layer {
    Config config;
    int layer = -1;
    bool full_indexer = false;
    int device = 0;
    DenseTensor attn_norm, ffn_norm, q_a_norm, kv_a_norm, index_k_norm, index_k_bias;
    std::vector<std::shared_ptr<const tyloi::Tensor>> input_proj, q_proj;
    std::shared_ptr<const tyloi::Tensor> embed_q, unembed_out;
    std::shared_ptr<const tyloi::Tensor> o_proj;
    Ffn ffn;
};

struct Model {
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    CudaModelPlan plan;
    Config config;
    tyloi::models::CausalLmMetadata metadata;
    std::shared_ptr<const tyloi::Tensor> embed, lm_head;
#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<const ::tyloi::Tensor> output_norm;
#else
    DenseTensor output_norm;
#endif
    RopeCache rope;
    std::unordered_map<int, RopeCache> device_ropes;
    std::vector<Layer> layers;
};
} // namespace tyloi::cuda::glm_dsa
