#pragma once

#include "engine/models/gemma4/config.h"
#include "engine/models/common/causal_metadata.h"
#include "engine/models/common/capabilities.h"
#include "engine/backends/cuda/ops/attention/full.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/weights/moe/experts.h"
#include "engine/models/source.h"

#include <memory>
#include <vector>

namespace mfq::cuda::gemma4 {

using Tensor = mfq_tensor_backend::Tensor;
using Config = mfq::models::gemma4::Config;

struct Layer {
    FullAttention attention;
    Tensor ffn_norm;
    Ffn ffn;
    bool gemma4_moe = false;
    Tensor attn_post_norm, ffn_post_norm, ffn_post_norm_1, ffn_pre_norm_2, ffn_post_norm_2;
    Tensor layer_scale;
    Experts gemma_moe_gate_up, gemma_moe_down;
    Tensor gemma_router, gemma_router_norm_scale, gemma_expert_scale;
    int gemma_top_k = 0;
    int device = 0;
};

struct Model {
    std::shared_ptr<const mfq::ModelSource> source;
    mfq::ModelGraph graph;
    CudaModelPlan plan;
    Config config;
    mfq::models::CausalLmMetadata metadata;
    double embed_scale = 1.0;
    QuantLinear embed, lm_head;
    Tensor output_norm;
    std::vector<Layer> layers;
};

} // namespace mfq::cuda::gemma4
