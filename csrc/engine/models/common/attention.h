#pragma once

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/ops.h"
#else
#include "engine/backends/cuda/ops/attention/attention.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#endif

#include <array>
#include <cstdint>
#include <memory>
#include <optional>

namespace tyloi::cuda {

struct AttentionStep {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    tyloi::DenseTensor positions;
    std::optional<tyloi::DenseTensor> sequence_lengths;
    std::optional<tyloi::DenseTensor> cache_positions;
    std::optional<tyloi::DenseTensor> attention_mask;
#else
    ::tyloi::DenseTensor positions;
    MfqOptional<::tyloi::DenseTensor> sequence_lengths = mfq_nullopt;
    MfqOptional<::tyloi::DenseTensor> cache_positions = mfq_nullopt;
    MfqOptional<::tyloi::DenseTensor> attention_mask = mfq_nullopt;
#endif
    std::int64_t cache_position = 0;
    std::int64_t planned_kv_length = 0;
    std::int64_t decode_attention_parts = 0;
};

struct FullAttention {
    int layer = -1;
    bool sliding = false;
    bool value_equals_key = false;
    std::int64_t attention_heads = 0;
    std::int64_t kv_heads = 0;
    std::int64_t attention_head_dim = 0;
    std::int64_t attention_rotary_dim = 0;
    std::int64_t attention_window = 0;
    std::int64_t max_position_embeddings = 0;
    double attention_scale = 0.0;
    double rms_norm_eps = 1e-6;
    double norm_weight_offset = 1.0;
    bool official_bf16 = false;
    // NoPE GQA: skip rotary operations while keeping the KV cache normal.
    bool no_rope = false;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    tyloi::ops::RopeTable attention_rope;
    tyloi::DenseTensor attn_norm, q_norm, k_norm, v_norm;
    std::shared_ptr<const QuantLinearGroup> qkv;
#else
    RopeCache attention_rope;
    ::tyloi::DenseTensor attn_norm, q_norm, k_norm, v_norm;
    QuantLinearGroup qkv;
#endif
    bool attention_output_gate = false;
    bool split_q_kv_projections = false;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    std::shared_ptr<const QuantLinear> q_projection;
    std::shared_ptr<const QuantLinear> k_projection;
    std::shared_ptr<const QuantLinear> v_projection;
    std::shared_ptr<const QuantLinear> o;
#else
    QuantLinear q_projection;
    QuantLinear k_projection;
    QuantLinear v_projection;
    QuantLinear o;
#endif
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    tyloi::state::KvQuantization kv_quantization;
    std::array<tyloi::DenseTensor, 2> forward(
        tyloi::DenseTensor input, const AttentionStep& step,
        tyloi::state::KvView cache, const tyloi::ops::RopeTable& rope,
        tyloi::ops::Trace* trace = nullptr) const;
#else
    tyloi::cuda::KvQuantization kv_quantization;
    std::array<::tyloi::DenseTensor, 2> forward(
        const Projector& project, CudaProfiler& profiler, const AttentionOptions& options,
        ::tyloi::DenseTensor input,
        const AttentionStep& step,
        AttentionKvView cache,
        AttentionScratch& scratch,
        const RopeCache& rope,
        Trace* trace = nullptr) const;
#endif
};

} // namespace tyloi::cuda
