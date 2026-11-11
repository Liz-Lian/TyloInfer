#pragma once

#include "engine/models/gemma4/config.h"
#include "engine/models/common/attention.h"
#include "engine/models/common/mlp.h"
#include "engine/models/source.h"
#include "engine/models/input.h"
#include "engine/models/causal_model.h"

#include <memory>
#include <utility>
#include <string_view>
#include <vector>

namespace tyloi::cuda::gemma4 {

using Config = tyloi::models::gemma4::Config;

class State;

struct Layer {
    FullAttention attention;
    ::tyloi::Tensor ffn_norm;
    tyloi::models::Mlp ffn;
    ::tyloi::Tensor attn_post_norm, ffn_post_norm, ffn_post_norm_1, ffn_pre_norm_2, ffn_post_norm_2;
    ::tyloi::Tensor layer_scale;
    ::tyloi::Tensor gemma_router, gemma_router_norm_scale, gemma_expert_scale;
    ::tyloi::Tensor gemma_moe_gate_up, gemma_moe_down;
    bool gemma4_moe = false;
    int gemma_top_k = 0;
    int device = 0;
};

/** Shared Gemma4 weights and configuration; each request owns its own State. */
struct Model : public tyloi::models::CausalModel {
    /** Return model dimensions and decode capabilities. */
    tyloi::models::Info info() const override;
    /** Return the model graph's supported capabilities. */
    MfqModelCapabilities capabilities() const override;
    /** Return model capability metrics. */
    tyloi::engine::Metrics metrics() const override;
    using Options = tyloi::models::ForwardOptions;
    /** Create a request state; this model must outlive it and stay at the same address. */
    std::unique_ptr<tyloi::models::State> make_state() const override;
    /** Reject multimodal preparation, which Gemma4 does not support here. */
    tyloi::StepSequence<tyloi::models::Prepared> prepare(const std::vector<std::int64_t>& tokens,
        const MfqMultimodalInput& media, tyloi::models::State& state) const override;
    /** Embed token IDs on the selected primary device. */
    ::tyloi::Tensor embedding(const tyloi::Tensor& ids, int primary = 0) const override;
    /** Apply output normalization to the hidden states. */
    ::tyloi::Tensor normalize(const tyloi::Tensor& hidden, std::int64_t batch, std::int64_t tokens) const override;
    /** Apply the configured final logit soft cap. */
    ::tyloi::Tensor softcap(const tyloi::Tensor& logits) const override;
    /** Project hidden states to optionally capped logits. */
    ::tyloi::Tensor logits(const tyloi::Tensor& hidden, bool capped = true) const override;
    /** Run the model with its creating request State, rejecting a foreign State before mutation. */
    ::tyloi::Tensor forward(tyloi::models::Input input, tyloi::models::State& state,
                        const Options& options = {}) const override;
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    Config config;
    double embed_scale = 1.0;
    ::tyloi::Tensor embed, lm_head;
    ::tyloi::Tensor output_norm;
    std::vector<Layer> layers;
};

/** Read and validate Gemma4 metadata before loading its weights. */
Model read(std::shared_ptr<const tyloi::ModelSource>, std::string_view definition, std::int64_t context);

} // namespace tyloi::cuda::gemma4
