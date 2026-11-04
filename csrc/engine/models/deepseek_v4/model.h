#pragma once

#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/backends/cuda/ops/rope/tail.h"
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
    DenseTensor ape, norm;

    std::vector<DenseTensor> project(DenseTensor x, int64_t batch, int64_t tokens) const;
    void update(PoolState&, DenseTensor kv, DenseTensor gate, DenseTensor length, const TailRope&) const;
    int64_t prefill(PoolState&, const DenseTensor& kv, const DenseTensor& gate, const TailRope&) const;
};

struct Layer {
    int layer = -1, device = 0;
    int64_t max_positions = 0, compress_ratio = 0, hidden_size = 4096, heads = 64;
    int64_t head_dim = 512, groups = 8, hc_mult = 4, hc_iterations = 20;
    double eps = 1e-6, hc_eps = 1e-6;
    DenseTensor attn_norm, ffn_norm, q_a_norm, kv_norm, sinks;
    DenseTensor hc_attn_fn, hc_attn_scale, hc_attn_base, hc_ffn_fn, hc_ffn_scale, hc_ffn_base;
    QuantLinear q_a, q_b, kv, output_a, output_b;
    Compressor compressor, indexer_compressor;
    QuantLinear indexer_q;
    DenseTensor indexer_weight;
    Ffn ffn;
    TailRope attention_rope;
};

struct OutputHeadWeights { DenseTensor function, scale, base; };

struct Model {
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    CudaModelPlan plan;
    Config config;
    tyloi::models::CausalLmMetadata metadata;
    QuantLinear embed, lm_head;
    DenseTensor output_norm;
    OutputHeadWeights output_head;
    std::vector<Layer> layers;
};
} // namespace tyloi::cuda::deepseek_v4
