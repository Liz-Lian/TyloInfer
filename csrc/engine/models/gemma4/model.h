#pragma once

#include "engine/models/gemma4/config.h"
#include "engine/models/common/capabilities.h"
#include "engine/models/common/attention.h"
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/models/common/mlp.h"
#else
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/weights/moe/experts.h"
#endif
#include "engine/weights/weights.h"
#include "engine/models/source.h"
#include "engine/models/options.h"
#include "engine/models/input.h"
#include "engine/models/causal_model.h"

#include <memory>
#include <string_view>
#include <vector>

namespace tyloi::cuda::gemma4 {

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
using Tensor = tyloi::Tensor;
#else
using Tensor = ::tyloi::Tensor;
#endif
using Config = tyloi::models::gemma4::Config;

class State;

struct Layer {
    FullAttention attention;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    tyloi::Tensor ffn_norm;
    tyloi::models::Mlp ffn;
    tyloi::Tensor attn_post_norm, ffn_post_norm, ffn_post_norm_1, ffn_pre_norm_2, ffn_post_norm_2;
    tyloi::Tensor layer_scale;
    std::shared_ptr<const tyloi::weights::Weights> gemma_moe_gate_up, gemma_moe_down;
    tyloi::Tensor gemma_router, gemma_router_norm_scale, gemma_expert_scale;
#else
    Tensor ffn_norm;
    Ffn ffn;
    Tensor attn_post_norm, ffn_post_norm, ffn_post_norm_1, ffn_pre_norm_2, ffn_post_norm_2;
    Tensor layer_scale;
    Experts gemma_moe_gate_up, gemma_moe_down;
    Tensor gemma_router, gemma_router_norm_scale, gemma_expert_scale;
#endif
    bool gemma4_moe = false;
    int gemma_top_k = 0;
    int device = 0;
};

struct Model
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    : public tyloi::models::CausalModel
#endif
{
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    tyloi::models::Info info() const override;
    MfqModelCapabilities capabilities() const override;
    tyloi::engine::Metrics metrics() const override;
#else
    tyloi::models::Info info() const;
    MfqModelCapabilities capabilities() const;
    tyloi::engine::Metrics metrics() const;
#endif
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    using Options = tyloi::models::ForwardOptions;
    std::unique_ptr<tyloi::models::State> make_state() const override;
    tyloi::StepSequence<tyloi::models::Prepared> prepare(const std::vector<std::int64_t>& tokens,
        const MfqMultimodalInput& media, tyloi::models::State& state) const override;
    tyloi::Tensor embedding(tyloi::Tensor ids, int primary = 0) const override;
    tyloi::Tensor normalize(tyloi::Tensor hidden, std::int64_t batch, std::int64_t tokens) const override;
    tyloi::Tensor softcap(tyloi::Tensor logits) const override;
    tyloi::Tensor logits(tyloi::Tensor hidden, bool capped = true) const override;
    tyloi::Tensor forward(tyloi::models::Input input, tyloi::models::State& state,
                        const Options& options = {}) const override;
#endif
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    CudaModelPlan plan;
    Config config;
    double embed_scale = 1.0;
    std::shared_ptr<const tyloi::weights::Weights> embed, lm_head;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    tyloi::Tensor output_norm;
#else
    Tensor output_norm;
#endif
    std::vector<Layer> layers;
};

Model read(std::shared_ptr<const tyloi::ModelSource>, std::string_view definition, std::int64_t context);

} // namespace tyloi::cuda::gemma4
