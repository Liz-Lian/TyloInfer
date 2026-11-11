#pragma once

#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/models/common/attention.h"
#include "engine/backends/cuda/ops/projection/quant_linear.h"
#include "engine/models/nemotron_h/config.h"
#include "engine/models/metadata.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/models/causal_model.h"
#endif
#include "engine/models/common/capabilities.h"
#include "engine/models/source.h"

#include <array>
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

namespace tyloi::cuda::nemotron_h {

using DenseTensor = ::tyloi::DenseTensor;
using Config = tyloi::models::nemotron_h::Config;
struct Components;

struct Mamba {
    int layer = -1;
    int64_t mamba_heads = 0, mamba_head_dim = 0, ssm_state_size = 0;
    int64_t mamba_groups = 0, conv_kernel = 0;
    double rms_norm_eps = 1e-6, time_step_limit_lo = 0.0, time_step_limit_hi = 0.0;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor mixer_norm;
#else
    DenseTensor mixer_norm;
#endif
    ::tyloi::Tensor in_proj;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor conv_weight, conv_bias, a_log, d, dt_bias, scan_norm;
#else
    DenseTensor conv_weight, conv_bias, a_log, d, dt_bias, scan_norm;
#endif
    ::tyloi::Tensor out_proj;
};

struct Moe {
    int layer = -1;
    double rms_norm_eps = 1e-6;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor ffn_norm;
#else
    DenseTensor ffn_norm;
#endif
    Ffn ffn;
};

struct Layer {
    std::variant<Mamba, FullAttention, Moe> weights;
    int device = 0;
};

struct Model
#ifdef TYLOINFER_NATIVE_CUDA
    : public tyloi::models::CausalModel
#endif
{
#ifdef TYLOINFER_NATIVE_CUDA
    /** Create an empty model before loading weights and components. */
    Model();
    ~Model() override;
    /** Transfer model ownership before creating request state. */
    Model(Model&&) noexcept;
    /** Replace model ownership before creating request state. */
    Model& operator=(Model&&) noexcept;
    /** Return model dimensions and graph policy. */
    tyloi::models::Info info() const override;
    /** Return capabilities provided by the assembled components. */
    MfqModelCapabilities capabilities() const override;
    /** Return assembled component and resource metrics. */
    tyloi::engine::Metrics metrics() const override;
    /** Borrow the assembled predictor, if present. */
    tyloi::models::Mtp* mtp() const noexcept override;
    /** Create independent request state and predictor state. */
    std::unique_ptr<tyloi::models::State> make_state() const override;
    /** Prepare supported RADIO media for the owning request. */
    tyloi::StepSequence<tyloi::models::Prepared> prepare(const std::vector<std::int64_t>&,
        const MfqMultimodalInput&, tyloi::models::State&) const override;
    /** Embed borrowed token IDs on the selected device. */
    ::tyloi::Tensor embedding(const tyloi::Tensor& ids, int primary = 0) const override;
    /** Normalize output hidden states. */
    ::tyloi::Tensor normalize(const tyloi::Tensor& hidden, std::int64_t batch, std::int64_t tokens) const override;
    /** Apply the configured final logit softcap. */
    ::tyloi::Tensor softcap(const tyloi::Tensor& logits) const override;
    /** Project hidden states to optionally capped logits. */
    ::tyloi::Tensor logits(const tyloi::Tensor& hidden, bool capped = true) const override;
    /** Execute against state created by this model. */
    ::tyloi::Tensor forward(tyloi::models::Input input, tyloi::models::State& state,
        const tyloi::models::ForwardOptions& options = {}) const override;
    /** Optional predictor and vision components owned by this model. */
    std::unique_ptr<Components> parts;
    /** Primary device used for media preparation. */
    int primary = 0;
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
    std::vector<Layer> layers;
};


} // namespace tyloi::cuda::nemotron_h
