#pragma once

#include "engine/models/glm_dsa/config.h"
#include "engine/models/metadata.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/models/causal_model.h"
#endif
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
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor attn_norm, ffn_norm, q_a_norm, kv_a_norm, index_k_norm, index_k_bias;
#else
    DenseTensor attn_norm, ffn_norm, q_a_norm, kv_a_norm, index_k_norm, index_k_bias;
#endif
    std::vector<::tyloi::Tensor> input_proj, q_proj;
    ::tyloi::Tensor embed_q, unembed_out;
    ::tyloi::Tensor o_proj;
    Ffn ffn;
};

struct Model
#ifdef TYLOINFER_NATIVE_CUDA
    : tyloi::models::CausalModel
#endif
{
#ifdef TYLOINFER_NATIVE_CUDA
    /** Return model dimensions and execution capabilities. */
    tyloi::models::Info info() const override;
    /** Return the capabilities of the assembled text model. */
    MfqModelCapabilities capabilities() const override;
    /** Return model capability and resource metrics. */
    tyloi::engine::Metrics metrics() const override;
    /** Create independent state owned by this model. */
    std::unique_ptr<tyloi::models::State> make_state() const override;
    /** Reject media input because this model has no multimodal component. */
    tyloi::StepSequence<tyloi::models::Prepared> prepare(const std::vector<std::int64_t>&,
        const MfqMultimodalInput&, tyloi::models::State&) const override;
    /** Embed borrowed token IDs on the requested primary device. */
    ::tyloi::Tensor embedding(const ::tyloi::Tensor& ids, int primary = 0) const override;
    /** Normalize output hidden states. */
    ::tyloi::Tensor normalize(const ::tyloi::Tensor& hidden, std::int64_t batch, std::int64_t tokens) const override;
    /** Apply the configured final logit softcap. */
    ::tyloi::Tensor softcap(const ::tyloi::Tensor& logits) const override;
    /** Project hidden states to logits, optionally applying the softcap. */
    ::tyloi::Tensor logits(const ::tyloi::Tensor& hidden, bool capped = true) const override;
    /** Run a model step against state created by this model. */
    ::tyloi::Tensor forward(tyloi::models::Input input, tyloi::models::State& state,
        const tyloi::models::ForwardOptions& options = {}) const override;
#endif
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    CudaModelPlan plan;
    Config config;
    tyloi::models::CausalLmMetadata metadata;
    ::tyloi::Tensor embed, lm_head;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor output_norm;
#else
    DenseTensor output_norm;
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::ops::RopeTable rope;
    std::unordered_map<int, tyloi::ops::RopeTable> device_ropes;
#else
    RopeCache rope;
    std::unordered_map<int, RopeCache> device_ropes;
#endif
    std::vector<Layer> layers;
};
} // namespace tyloi::cuda::glm_dsa
