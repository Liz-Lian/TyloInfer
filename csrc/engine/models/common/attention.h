#pragma once

#include "engine/ops.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>

namespace tyloi::cuda {

struct AttentionStep {
    ::tyloi::Tensor positions;
    ::tyloi::Tensor sequence_lengths;
    ::tyloi::Tensor cache_positions;
    std::int64_t cache_position = 0;
    std::int64_t planned_kv_length = 0;
    std::int64_t decode_attention_parts = 0;
    /** Optional key-validity mask covering the visible cache. */
    ::tyloi::Tensor attention_mask;
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
    tyloi::ops::RopeTable attention_rope;
    ::tyloi::Tensor attn_norm, q_norm, k_norm, v_norm;
    std::vector<::tyloi::Tensor> qkv;
    bool attention_output_gate = false;
    bool split_q_kv_projections = false;
    ::tyloi::Tensor q_projection, k_projection, v_projection, o;
    tyloi::state::KvQuantization kv_quantization;
    std::array<::tyloi::Tensor, 2> forward(
        const tyloi::Tensor& input, const AttentionStep& step,
        tyloi::state::KvView cache, const tyloi::ops::RopeTable& rope,
        tyloi::ops::Trace* trace = nullptr) const;
};

} // namespace tyloi::cuda
