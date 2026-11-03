#pragma once

#include "engine/models/gemma4/config.h"
#include "engine/models/common/capabilities.h"
#include "engine/models/common/attention.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/weights/moe/experts.h"
#include "engine/weights/matrix.h"
#include "engine/models/source.h"
#include "engine/models/options.h"
#include "engine/models/input.h"
#include "engine/models/causal_model.h"

#include <memory>
#include <string_view>
#include <vector>

namespace mfq::cuda::gemma4 {

using Tensor = mfq_tensor_backend::Tensor;
using Config = mfq::models::gemma4::Config;

class State;

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

struct Model
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    : public mfq::models::CausalModel
#endif
{
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    mfq::models::Info info() const override;
#else
    mfq::models::Info info() const;
#endif
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    using Options = mfq::models::ForwardOptions;
    std::unique_ptr<mfq::models::State> make_state() const override;
    mfq::StepSequence<mfq::models::Prepared> prepare(const std::vector<std::int64_t>& tokens,
        const MfqMultimodalInput& media, mfq::models::State& state) const override;
    mfq::Tensor embedding(mfq::Tensor ids, int primary = 0) const override;
    mfq::Tensor normalize(mfq::Tensor hidden, std::int64_t batch, std::int64_t tokens) const override;
    mfq::Tensor softcap(mfq::Tensor logits) const override;
    mfq::Tensor logits(mfq::Tensor hidden, bool capped = true, const Options& options = {}) const override;
    mfq::Tensor forward(mfq::models::Input input, mfq::models::State& state,
                        const Options& options = {}) const override;
#endif
    std::shared_ptr<const mfq::ModelSource> source;
    mfq::ModelGraph graph;
    CudaModelPlan plan;
    Config config;
    double embed_scale = 1.0;
    std::shared_ptr<const mfq::weights::Matrix> embed, lm_head;
    Tensor output_norm;
    std::vector<Layer> layers;
};

Model read(std::shared_ptr<const mfq::ModelSource>, std::string_view definition, std::int64_t context);

} // namespace mfq::cuda::gemma4
