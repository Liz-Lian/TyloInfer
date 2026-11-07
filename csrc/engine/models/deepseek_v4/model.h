#pragma once

#include "engine/backends/cuda/ops/ffn/ffn.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/ops.h"
#else
#include "engine/backends/cuda/ops/rope/tail.h"
#endif
#include "engine/models/common/causal_metadata.h"
#include "engine/models/common/capabilities.h"
#include "engine/models/deepseek_v4/config.h"
#include "engine/models/source.h"

#include <memory>
#include <vector>

namespace tyloi::cuda::deepseek_v4 {
using DenseTensor = ::tyloi::DenseTensor;
using Config = tyloi::models::deepseek_v4::Config;
struct PoolState;

struct Compressor {
    int64_t ratio = 0, head_dim = 0, cache_quant_mode = 0;
    bool overlap = false;
    DenseLinearGroup projection;
#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<const tyloi::Tensor> ape, norm;
#else
    DenseTensor ape, norm;
#endif

    std::vector<DenseTensor> project(DenseTensor x, int64_t batch, int64_t tokens) const;
#ifdef TYLOINFER_NATIVE_CUDA
    void update(PoolState&, DenseTensor kv, DenseTensor gate, DenseTensor length, const tyloi::ops::TailRopeTable&) const;
    int64_t prefill(PoolState&, const DenseTensor& kv, const DenseTensor& gate, const tyloi::ops::TailRopeTable&) const;
#else
    void update(PoolState&, DenseTensor kv, DenseTensor gate, DenseTensor length, const TailRope&) const;
    int64_t prefill(PoolState&, const DenseTensor& kv, const DenseTensor& gate, const TailRope&) const;
#endif
};

struct Layer {
    int layer = -1, device = 0;
    int64_t max_positions = 0, compress_ratio = 0, hidden_size = 4096, heads = 64;
    int64_t head_dim = 512, groups = 8, hc_mult = 4, hc_iterations = 20;
    double eps = 1e-6, hc_eps = 1e-6;
#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<const tyloi::Tensor> attn_norm, ffn_norm, q_a_norm, kv_norm, sinks;
    std::shared_ptr<const tyloi::Tensor> hc_attn_fn, hc_attn_scale, hc_attn_base, hc_ffn_fn, hc_ffn_scale, hc_ffn_base;
#else
    DenseTensor attn_norm, ffn_norm, q_a_norm, kv_norm, sinks;
    DenseTensor hc_attn_fn, hc_attn_scale, hc_attn_base, hc_ffn_fn, hc_ffn_scale, hc_ffn_base;
#endif
    std::shared_ptr<const tyloi::Tensor> q_a, q_b, kv, output_a, output_b;
    Compressor compressor, indexer_compressor;
    std::shared_ptr<const tyloi::Tensor> indexer_q;
#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<const tyloi::Tensor> indexer_weight;
#else
    DenseTensor indexer_weight;
#endif
    Ffn ffn;
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::ops::TailRopeTable attention_rope;
#else
    TailRope attention_rope;
#endif
};

struct OutputHeadWeights {
#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<const tyloi::Tensor> function, scale, base;
#else
    DenseTensor function, scale, base;
#endif
};

struct Model {
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    CudaModelPlan plan;
    Config config;
    tyloi::models::CausalLmMetadata metadata;
    std::shared_ptr<const tyloi::Tensor> embed, lm_head;
#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<const tyloi::Tensor> output_norm;
#else
    DenseTensor output_norm;
#endif
    OutputHeadWeights output_head;
    std::vector<Layer> layers;
};
} // namespace tyloi::cuda::deepseek_v4
