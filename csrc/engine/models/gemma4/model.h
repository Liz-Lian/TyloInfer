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

struct Model : public tyloi::models::CausalModel {
    tyloi::models::Info info() const override;
    MfqModelCapabilities capabilities() const override;
    tyloi::engine::Metrics metrics() const override;
    using Options = tyloi::models::ForwardOptions;
    std::unique_ptr<tyloi::models::State> make_state() const override;
    tyloi::StepSequence<tyloi::models::Prepared> prepare(const std::vector<std::int64_t>& tokens,
        const MfqMultimodalInput& media, tyloi::models::State& state) const override;
    ::tyloi::Tensor embedding(const tyloi::Tensor& ids, int primary = 0) const override;
    ::tyloi::Tensor normalize(const tyloi::Tensor& hidden, std::int64_t batch, std::int64_t tokens) const override;
    ::tyloi::Tensor softcap(const tyloi::Tensor& logits) const override;
    ::tyloi::Tensor logits(const tyloi::Tensor& hidden, bool capped = true) const override;
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

Model read(std::shared_ptr<const tyloi::ModelSource>, std::string_view definition, std::int64_t context);

} // namespace tyloi::cuda::gemma4
