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

namespace mfq::cuda::glm_dsa {
using Tensor = mfq_tensor_backend::Tensor;
using Config = mfq::models::glm_dsa::Config;

struct Layer {
    Config config;
    int layer = -1;
    bool full_indexer = false;
    int device = 0;
    Tensor attn_norm, ffn_norm, q_a_norm, kv_a_norm, index_k_norm, index_k_bias;
    QuantLinearGroup input_proj, q_proj;
    Experts embed_q, unembed_out;
    QuantLinear o_proj;
    Ffn ffn;
};

struct Model {
    std::shared_ptr<const mfq::ModelSource> source;
    mfq::ModelGraph graph;
    CudaModelPlan plan;
    Config config;
    mfq::models::CausalLmMetadata metadata;
    QuantLinear embed, lm_head;
    Tensor output_norm;
    RopeCache rope;
    std::unordered_map<int, RopeCache> device_ropes;
    std::vector<Layer> layers;
};
} // namespace mfq::cuda::glm_dsa
