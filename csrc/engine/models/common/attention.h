#pragma once

#include "engine/backends/cuda/ops/attention/attention.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/ops.h"
#endif

#include <array>

namespace mfq::cuda {

struct AttentionStep {
    mfq_tensor_backend::Tensor positions;
    MfqOptional<mfq_tensor_backend::Tensor> sequence_lengths = mfq_nullopt;
    MfqOptional<mfq_tensor_backend::Tensor> cache_positions = mfq_nullopt;
    MfqOptional<mfq_tensor_backend::Tensor> attention_mask = mfq_nullopt;
    int64_t cache_position = 0;
    int64_t planned_kv_length = 0;
    int64_t decode_attention_parts = 0;
};

struct FullAttention {
    int layer = -1;
    bool sliding = false;
    bool value_equals_key = false;
    int64_t attention_heads = 0;
    int64_t kv_heads = 0;
    int64_t attention_head_dim = 0;
    int64_t attention_rotary_dim = 0;
    int64_t attention_window = 0;
    int64_t max_position_embeddings = 0;
    double attention_scale = 0.0;
    double rms_norm_eps = 1e-6;
    double norm_weight_offset = 1.0;
    bool official_bf16 = false;
    // NoPE GQA: skip rotary operations while keeping the KV cache normal.
    bool no_rope = false;
    RopeCache attention_rope;

    mfq_tensor_backend::Tensor attn_norm;
    mfq_tensor_backend::Tensor q_norm;
    mfq_tensor_backend::Tensor k_norm;
    mfq_tensor_backend::Tensor v_norm;
    QuantLinearGroup qkv;
    bool attention_output_gate = false;
    bool split_q_kv_projections = false;
    QuantLinear q_projection;
    QuantLinear k_projection;
    QuantLinear v_projection;
    QuantLinear o;
    mfq::cuda::KvQuantization kv_quantization;

    static constexpr int64_t kDecodeAttentionMaxParts = AttentionScratch::kMaxParts;

    std::array<mfq_tensor_backend::Tensor, 2> forward(
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        mfq_tensor_backend::Tensor input, const AttentionStep& step,
        AttentionKvView cache, const RopeCache& rope,
        const mfq::ops::ProjectionOptions& projection = {}, Trace* trace = nullptr) const;
#else
        const Projector& project, CudaProfiler& profiler, const AttentionOptions& options,
        mfq_tensor_backend::Tensor input,
        const AttentionStep& step,
        AttentionKvView cache,
        AttentionScratch& scratch,
        const RopeCache& rope,
        Trace* trace = nullptr) const;
#endif
};

} // namespace mfq::cuda
