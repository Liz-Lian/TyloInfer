#pragma once

#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/models/common/attention.h"
#include "engine/backends/cuda/ops/projection/quant_linear.h"
#include "engine/models/nemotron_h/config.h"
#include "engine/models/common/causal_metadata.h"
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

struct Mamba {
    int layer = -1;
    int64_t mamba_heads = 0, mamba_head_dim = 0, ssm_state_size = 0;
    int64_t mamba_groups = 0, conv_kernel = 0;
    double rms_norm_eps = 1e-6, time_step_limit_lo = 0.0, time_step_limit_hi = 0.0;
    DenseTensor mixer_norm;
    std::shared_ptr<const ::tyloi::Tensor> in_proj;
    DenseTensor conv_weight, conv_bias, a_log, d, dt_bias, scan_norm;
    std::shared_ptr<const ::tyloi::Tensor> out_proj;
};

struct Moe {
    int layer = -1;
    double rms_norm_eps = 1e-6;
    DenseTensor ffn_norm;
    Ffn ffn;
};

struct Layer {
    std::variant<Mamba, FullAttention, Moe> weights;
    int device = 0;
};

struct Model {
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    CudaModelPlan plan;
    Config config;
    tyloi::models::CausalLmMetadata metadata;
    std::shared_ptr<const ::tyloi::Tensor> embed, lm_head;
#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<const ::tyloi::Tensor> output_norm;
#else
    DenseTensor output_norm;
#endif
    std::vector<Layer> layers;
#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<tyloi::Tensor> embedding(const tyloi::Tensor& ids, int primary = 0) const;
#endif
};


} // namespace tyloi::cuda::nemotron_h
