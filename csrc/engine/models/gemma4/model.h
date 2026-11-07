#pragma once

#include "engine/models/gemma4/config.h"
#include "engine/models/common/capabilities.h"
#include "engine/models/common/attention.h"
#ifdef TYLOINFER_NATIVE_CUDA
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

#ifndef TYLOINFER_NATIVE_CUDA
using DenseTensor = ::tyloi::DenseTensor;
#endif
using Config = tyloi::models::gemma4::Config;

class State;

struct Layer {
    FullAttention attention;
#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<const tyloi::Tensor> ffn_norm;
    tyloi::models::Mlp ffn;
    std::shared_ptr<const tyloi::Tensor> attn_post_norm, ffn_post_norm, ffn_post_norm_1, ffn_pre_norm_2, ffn_post_norm_2;
    std::shared_ptr<const tyloi::Tensor> layer_scale;
    std::shared_ptr<const tyloi::Tensor> gemma_router, gemma_router_norm_scale, gemma_expert_scale;
#else
    DenseTensor ffn_norm;
    Ffn ffn;
    DenseTensor attn_post_norm, ffn_post_norm, ffn_post_norm_1, ffn_pre_norm_2, ffn_post_norm_2;
    DenseTensor layer_scale;
    DenseTensor gemma_router, gemma_router_norm_scale, gemma_expert_scale;
#endif
    std::shared_ptr<const tyloi::Tensor> gemma_moe_gate_up, gemma_moe_down;
    bool gemma4_moe = false;
    int gemma_top_k = 0;
    int device = 0;
};

struct Model
#ifdef TYLOINFER_NATIVE_CUDA
    : public tyloi::models::CausalModel
#endif
{
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::models::Info info() const override;
    MfqModelCapabilities capabilities() const override;
    tyloi::engine::Metrics metrics() const override;
#else
    tyloi::models::Info info() const;
    MfqModelCapabilities capabilities() const;
    tyloi::engine::Metrics metrics() const;
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    using Options = tyloi::models::ForwardOptions;
    std::unique_ptr<tyloi::models::State> make_state() const override;
    tyloi::StepSequence<tyloi::models::Prepared> prepare(const std::vector<std::int64_t>& tokens,
        const MfqMultimodalInput& media, tyloi::models::State& state) const override;
    std::shared_ptr<tyloi::Tensor> embedding(const tyloi::Tensor& ids, int primary = 0) const override;
    std::shared_ptr<tyloi::Tensor> normalize(const tyloi::Tensor& hidden, std::int64_t batch, std::int64_t tokens) const override;
    std::shared_ptr<tyloi::Tensor> softcap(const tyloi::Tensor& logits) const override;
    std::shared_ptr<tyloi::Tensor> logits(const tyloi::Tensor& hidden, bool capped = true) const override;
    std::shared_ptr<tyloi::Tensor> forward(tyloi::models::Input input, tyloi::models::State& state,
                        const Options& options = {}) const override;
#endif
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    CudaModelPlan plan;
    Config config;
    double embed_scale = 1.0;
    std::shared_ptr<const ::tyloi::Tensor> embed, lm_head;
#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<const tyloi::Tensor> output_norm;
#else
    DenseTensor output_norm;
#endif
    std::vector<Layer> layers;
};

Model read(std::shared_ptr<const tyloi::ModelSource>, std::string_view definition, std::int64_t context);

} // namespace tyloi::cuda::gemma4
