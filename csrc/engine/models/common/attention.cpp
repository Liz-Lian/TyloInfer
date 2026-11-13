#include "engine/models/common/attention.h"
#include "engine/ops.h"

#include <stdexcept>
#include <utility>
#include <vector>


namespace tyloi::cuda {

std::array<::tyloi::Tensor, 2> FullAttention::forward(
        const tyloi::Tensor& input, const AttentionStep& step, tyloi::state::KvView cache,
        const tyloi::ops::RopeTable& rope, tyloi::ops::Trace* trace) const {
    if (!step.positions) throw std::runtime_error("attention positions are missing");
    const auto& pos = step.positions;
    const auto cache_pos = step.cache_position;
    const auto profile_device = input.device();
    auto measure = [&](const char* name, auto&& operation) {
        return tyloi::ops::measure(name, profile_device, std::forward<decltype(operation)>(operation));
    };
    const std::int64_t B = input.size(0), T = input.size(1);
    auto trace_qwen_stage = [&](const char* name, const tyloi::Tensor& value, int token_axis = 1) {
        if (!attention_output_gate || trace == nullptr) return;
        ::tyloi::Tensor ordered;
        if (token_axis != 1) {
            auto transposed = tyloi::ops::transpose(value, 1, token_axis);
            ordered = tyloi::ops::contiguous(transposed);
        }
        auto shaped = tyloi::ops::reshape(ordered ? ordered : value, std::array{B, T, std::int64_t{-1}});
        auto converted = tyloi::ops::cast(shaped, tyloi::ScalarType::float32);
        auto contiguous = tyloi::ops::contiguous(converted);
        trace->emplace_back(name, tyloi::ops::clone(contiguous));
    };
    ::tyloi::Tensor x;
    if (official_bf16 && input.scalar_type() != tyloi::ScalarType::bfloat16) {
        auto converted = tyloi::ops::cast(input, tyloi::ScalarType::bfloat16);
        x = tyloi::ops::contiguous(converted);
    } else {
        x = tyloi::ops::copy(input, input.device());
    }
    auto residual = x;
    const std::int64_t nh = attention_heads, nkh = kv_heads, hd = attention_head_dim;
    const std::int64_t attn_width = nh * hd;
    const auto& active_rope = attention_rope.cos && attention_rope.cos.defined() ? attention_rope : rope;
    const bool apply_rotary = !no_rope;
    auto xn = measure("full.attn_norm", [&]() {
        return tyloi::ops::rms_norm(x, attn_norm, rms_norm_eps, norm_weight_offset,
            official_bf16 ? tyloi::ScalarType::bfloat16 : tyloi::ScalarType::float32);
    });
    auto positions_cast = tyloi::ops::cast(step.cache_positions ? step.cache_positions : pos,
                                           tyloi::ScalarType::int64, x.device());
    auto write_positions = tyloi::ops::contiguous(positions_cast);
    if (!((write_positions.dim() == 1 && write_positions.numel() == T) ||
          (write_positions.dim() == 2 && write_positions.size(0) == B &&
           write_positions.size(1) == T))) {
        throw std::runtime_error("KV cache positions must have shape [tokens] or [batch,tokens]");
    }
    // A leading extent of three denotes semantic axes only in the single-image path.
    const bool grid_mrope_positions = B == 1 && step.cache_positions && pos.dim() == 2 && pos.size(0) == 3;
    trace_qwen_stage("qwen.attn_norm", xn);
    auto parts = measure("full.qkv", [&]() {
        if (!split_q_kv_projections) return tyloi::ops::project_group(xn, qkv);
        std::vector<::tyloi::Tensor> result;
        result.emplace_back(tyloi::ops::project(xn, q_projection));
        result.emplace_back(tyloi::ops::project(xn, k_projection));
        result.emplace_back(tyloi::ops::project(xn, v_projection));
        return result;
    });
    if (parts.size() != (value_equals_key ? 2u : 3u))
        throw std::runtime_error("attention projection group has the wrong output count");
    auto q_full = parts[0], k_full = parts[1];
    auto v_full = value_equals_key ? k_full : parts[2];
    trace_qwen_stage("qwen.q_projection", q_full);
    trace_qwen_stage("qwen.k_projection", k_full);
    trace_qwen_stage("qwen.v_projection", v_full);
    if (official_bf16) {
        auto q_cast = tyloi::ops::cast(q_full, tyloi::ScalarType::bfloat16);
        q_full = tyloi::ops::contiguous(q_cast);
        auto k_cast = tyloi::ops::cast(k_full, tyloi::ScalarType::bfloat16);
        k_full = tyloi::ops::contiguous(k_cast);
        auto v_cast = tyloi::ops::cast(v_full, tyloi::ScalarType::bfloat16);
        v_full = tyloi::ops::contiguous(v_cast);
    }
    ::tyloi::Tensor q_raw, q_gate;
    measure("full.qkv_view", [&]() {
        if (attention_output_gate) {
            auto qp = tyloi::ops::reshape(q_full, std::array{B, T, nh, hd * 2});
            auto chunks = tyloi::ops::chunk(qp, 2, -1);
            q_raw = chunks[0];
            q_gate = chunks[1];
        } else {
            q_raw = tyloi::ops::reshape(q_full, std::array{B, T, nh, hd});
        }
        return q_raw;
    });
    auto q = measure("full.q_view", [&]() {
        auto view = tyloi::ops::transpose(q_raw, 1, 2);
        return tyloi::ops::contiguous(view);
    });
    auto k = measure("full.k_view", [&]() {
        auto shaped = tyloi::ops::reshape(k_full, std::array{B, T, nkh, hd});
        auto view = tyloi::ops::transpose(shaped, 1, 2);
        return tyloi::ops::contiguous(view);
    });
    auto v = measure("full.v_view", [&]() {
        auto shaped = tyloi::ops::reshape(v_full, std::array{B, T, nkh, hd});
        auto view = tyloi::ops::transpose(shaped, 1, 2);
        return tyloi::ops::contiguous(view);
    });
    std::pair<::tyloi::Tensor, ::tyloi::Tensor> kv;
    bool prepared = false;
    if (apply_rotary && official_bf16 && (!v_norm || !v_norm.defined()) &&
        q_norm && q_norm.defined() && k_norm && k_norm.defined()) {
        auto result = tyloi::ops::norm_rope_kv(
            tyloi::ops::AttentionInput{q, k, v, pos, write_positions}, cache,
            active_rope, q_norm, k_norm, rms_norm_eps, norm_weight_offset);
        if (result) {
            q = std::move(result);
            prepared = true;
        }
    }
    if (!prepared) {
        auto normalized = tyloi::ops::qkv_norm({q, k, v},
            {q_norm, k_norm, v_norm}, rms_norm_eps, norm_weight_offset,
            official_bf16 ? tyloi::ScalarType::bfloat16 : tyloi::ScalarType::float32);
        q = normalized[0];
        k = normalized[1];
        v = normalized[2];
        if (apply_rotary && official_bf16) {
            auto result = tyloi::ops::rope_kv(
                tyloi::ops::AttentionInput{q, k, v, pos, write_positions}, cache, active_rope);
            if (result) {
                q = std::move(result);
                prepared = true;
            }
        }
        if (!prepared) {
            if (apply_rotary) {
                q = measure("full.q_rope", [&]() {
                    return tyloi::ops::rope(q, pos, active_rope, official_bf16, grid_mrope_positions);
                });
                k = measure("full.k_rope", [&]() {
                    return tyloi::ops::rope(k, pos, active_rope, official_bf16, grid_mrope_positions);
                });
            }
            kv = measure("full.kv_write", [&]() {
                return tyloi::ops::append_kv(cache, k, v, write_positions, cache_pos,
                                             !step.sequence_lengths);
            });
        }
    }
    if (prepared) {
        kv = {tyloi::ops::slice(cache.physical_key, 2, 0, cache_pos + T),
              tyloi::ops::slice(cache.physical_value, 2, 0, cache_pos + T)};
    }
    trace_qwen_stage("qwen.q_rope", q, 2);
    trace_qwen_stage("qwen.k_rope", k, 2);
    cache.visible_key = kv.first;
    cache.visible_value = kv.second;
    auto result = tyloi::ops::attention(
        tyloi::ops::AttentionInput{q, k, v, pos, write_positions,
            step.sequence_lengths, step.attention_mask, cache_pos,
            step.planned_kv_length, step.decode_attention_parts}, cache,
        tyloi::ops::AttentionConfig{official_bf16, sliding, attention_window, attention_scale});
    auto a = result.value;
    ::tyloi::Tensor oo;
    if (q_gate) trace_qwen_stage("qwen.attention", a, result.token_major ? 1 : 2);
    auto af = measure("full.attn_out_view", [&]() {
        if (!result.token_major) {
            auto transposed = tyloi::ops::transpose(a, 1, 2);
            a = tyloi::ops::contiguous(transposed);
        }
        return tyloi::ops::reshape(a, std::array{B, T, attn_width});
    });
    if (q_gate) {
        auto gf = measure("full.q_gate_view", [&]() {
            auto contiguous = tyloi::ops::contiguous(q_gate);
            return tyloi::ops::reshape(contiguous, std::array{B, T, attn_width});
        });
        oo = measure("full.o_proj_gate", [&]() { return tyloi::ops::sigmoid_project(af, gf, o); });
    } else {
        oo = measure("full.o_proj", [&]() {
            auto output = tyloi::ops::project(af, o);
            if (!official_bf16) return output;
            auto converted = tyloi::ops::cast(output, tyloi::ScalarType::bfloat16);
            return tyloi::ops::contiguous(converted);
        });
    }
    trace_qwen_stage("qwen.o_projection", oo);
    if (official_bf16) {
        auto converted = tyloi::ops::cast(oo, tyloi::ScalarType::bfloat16);
        oo = tyloi::ops::contiguous(converted);
    }
    return {std::move(residual), std::move(oo)};
}

} // namespace tyloi::cuda
