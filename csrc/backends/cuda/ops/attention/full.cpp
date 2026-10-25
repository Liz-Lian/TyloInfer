#include "full.h"
#include "kv.h"
#include "ops/norm/rms.h"
#include "engine/models/common/attention.h"
#include "engine/models/common/transformer_layer.h"
#include "mfq_cuda_activation_ops.h"
#include "mfq_cuda_attention_ops.h"
#include "mfq_cuda_norm_ops.h"

#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

using mfq_tensor_backend::indexing::Slice;

namespace mfq::cuda {

std::array<mfq_tensor_backend::Tensor, 2> FullAttention::forward(
        CudaExecutionContext& execution, mfq_tensor_backend::Tensor x,
        const AttentionStep& step, AttentionKvView cache,
        AttentionScratch& scratch, const RopeCache& rope) const {
    const auto& pos = step.positions;
    const auto& seq_len = step.sequence_lengths;
    const auto& cache_positions = step.cache_positions;
    const auto& attention_mask = step.attention_mask;
    const auto cache_pos = step.cache_position;
    const auto planned_kv_length = step.planned_kv_length;
    const auto decode_attention_parts = step.decode_attention_parts;
    auto &profiler = execution.profiler;
    int64_t B = x.size(0), T = x.size(1), H = x.size(2);
    auto trace_qwen_stage = [&](const char *name, const mfq_tensor_backend::Tensor &value,
                                int token_axis = 1) {
        if (!attention_output_gate || execution.gemma_stage_trace == nullptr ||
            layer != execution.gemma_trace_layer)
            return;
        auto ordered = token_axis == 1 ? value : value.transpose(1, token_axis).contiguous();
        trace_gemma_stage(execution, layer, name, ordered.reshape({B, T, -1}));
    };
    if (official_bf16 && x.scalar_type() != mfq_tensor_backend::kBFloat16) {
        x = x.to(mfq_tensor_backend::kBFloat16).contiguous();
    }
    const int64_t nh = attention_heads;
    const int64_t nkh = kv_heads;
    const int64_t hd = attention_head_dim;
    const int64_t attn_width = nh * hd;
    const RopeCache &active_rope = attention_rope.cos.defined() ? attention_rope : rope;
    const bool apply_rotary = !no_rope;
    if (x.is_cuda())
        scratch.split(B * nh, kDecodeAttentionMaxParts, hd, x.device());
    return mfq::models::attention_layer(
        x,
        [&](mfq_tensor_backend::Tensor x) {
            return profiler.measure("full.attn_norm", [&]() {
                return official_bf16
                           ? qwen_rms_norm_bf16(execution.config.minicpm_fused_bf16_rmsnorm, x.reshape({B * T, H}), attn_norm,
                                                rms_norm_eps, norm_weight_offset)
                                 .reshape({B, T, H})
                           : qwen_rms_norm(x.reshape({B * T, H}).to(mfq_tensor_backend::kFloat32),
                                           attn_norm, rms_norm_eps, norm_weight_offset)
                                 .reshape({B, T, H});
            });
        },
        [&](mfq_tensor_backend::Tensor xn) {
            auto write_positions =
                cache_positions.has_value()
                    ? cache_positions.value()
                          .to(x.device(), mfq_tensor_backend::kInt64)
                          .contiguous()
                    : pos.to(x.device(), mfq_tensor_backend::kInt64).contiguous();
            if (!((write_positions.dim() == 1 && write_positions.numel() == T) ||
                  (write_positions.dim() == 2 && write_positions.size(0) == B &&
                   write_positions.size(1) == T))) {
                throw std::runtime_error(
                    "KV cache positions must have shape [tokens] or [batch,tokens]");
            }
            // A leading extent of three is semantic axes only for the prepared
            // single-image path. B=3 ordinary batching must remain batch-major.
            const bool grid_mrope_positions =
                B == 1 && cache_positions.has_value() && pos.dim() == 2 && pos.size(0) == 3;
            const bool fused_qk_rope_kv =
                apply_rotary && official_bf16 && x.is_cuda() &&
                write_positions.dim() == 1 && pos.dim() == 1 &&
                T == 1 && !cache.paged() && !cache.ring && !cache.quantization.enabled() && !v_norm.defined() &&
                q_norm.defined() && k_norm.defined() && active_rope.rotary_dim == 128 &&
                active_rope.sections.numel() == 0 && nh == 32 && nkh == 8 && hd == 128 &&
                cache.dtype == mfq_tensor_backend::kBFloat16 &&
                execution.config.minicpm_fused_qk_norm_rope_kv;
            std::pair<mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor> kv;
            bool attention_token_major = false;
            return mfq::models::full_attention(
                std::move(xn),
                [&](mfq_tensor_backend::Tensor xn) {
                    trace_qwen_stage("qwen.attn_norm", xn);
                    auto parts = profiler.measure("full.qkv", [&]() {
                        if (!split_q_kv_projections) {
                            return qkv.forward(execution, xn);
                        }
                        std::vector<mfq_tensor_backend::Tensor> result;
                        result.reserve(3);
                        result.push_back(q_projection.forward(execution, xn));
                        result.push_back(k_projection.forward(execution, xn));
                        result.push_back(v_projection.forward(execution, xn));
                        return result;
                    });
                    if (parts.size() != (value_equals_key ? 2u : 3u)) {
                        throw std::runtime_error(
                            "attention projection group has the wrong output count");
                    }
                    auto q_full = parts[0], k_full = parts[1];
                    auto v_full = value_equals_key ? k_full : parts[2];
                    trace_qwen_stage("qwen.q_projection", q_full);
                    trace_qwen_stage("qwen.k_projection", k_full);
                    trace_qwen_stage("qwen.v_projection", v_full);
                    if (official_bf16) {
                        q_full = q_full.to(mfq_tensor_backend::kBFloat16).contiguous();
                        k_full = k_full.to(mfq_tensor_backend::kBFloat16).contiguous();
                        v_full = v_full.to(mfq_tensor_backend::kBFloat16).contiguous();
                    }
                    mfq_tensor_backend::Tensor q_raw, q_gate;
                    profiler.measure("full.qkv_view", [&]() {
                        if (attention_output_gate) {
                            auto qp = q_full.reshape({B, T, nh, hd * 2});
                            auto chunks = qp.chunk(2, -1);
                            q_raw = chunks[0];
                            q_gate = chunks[1];
                        } else {
                            q_raw = q_full.reshape({B, T, nh, hd});
                        }
                        return q_raw;
                    });
                    auto q = profiler.measure("full.q_view",
                                              [&]() { return q_raw.transpose(1, 2).contiguous(); });
                    auto k = profiler.measure("full.k_view", [&]() {
                        return k_full.reshape({B, T, nkh, hd}).transpose(1, 2).contiguous();
                    });
                    auto v = profiler.measure("full.v_view", [&]() {
                        return v_full.reshape({B, T, nkh, hd}).transpose(1, 2).contiguous();
                    });
                    return mfq::models::AttentionHeads<mfq_tensor_backend::Tensor>{
                        q, k, v,
                        q_gate.defined() ? std::optional<mfq_tensor_backend::Tensor>(q_gate)
                                         : std::nullopt};
                },
                [&](auto &heads) {
                    auto &q = heads.query;
                    auto &k = heads.key;
                    auto &v = heads.value;
                    if (!fused_qk_rope_kv)
                        return false;

                    q = profiler.measure("full.qk_norm_rope_kv_write", [&]() {
                        return minicpm_qk_norm_rope_cache_write_bf16_cuda(
                            q.contiguous(), k.contiguous(), v.contiguous(), q_norm, k_norm,
                            pos.contiguous().to(x.device(), mfq_tensor_backend::kInt64),
                            write_positions, active_rope.cos, active_rope.sin, cache.physical_key, cache.physical_value,
                            rms_norm_eps, norm_weight_offset);
                    });
                    kv = {cache.physical_key.index({Slice(), Slice(), Slice(0, cache_pos + T), Slice()}),
                          cache.physical_value.index({Slice(), Slice(), Slice(0, cache_pos + T), Slice()})};
                    return true;
                },
                [&](auto &heads) {
                    auto &q = heads.query;
                    auto &k = heads.key;
                    auto &v = heads.value;

                    const bool fused_bf16_norm =
                        official_bf16 && execution.config.minicpm_fused_bf16_rmsnorm;
                    if (fused_bf16_norm && q_norm.defined() && k_norm.defined() &&
                        q.scalar_type() == mfq_tensor_backend::kBFloat16 &&
                        k.scalar_type() == mfq_tensor_backend::kBFloat16) {
                        auto normalized = profiler.measure("full.qk_norm", [&]() {
                            return qwen_rms_norm_pair_bf16_cuda(q, k, q_norm, k_norm, rms_norm_eps,
                                                                norm_weight_offset);
                        });
                        q = normalized[0].reshape_as(q);
                        k = normalized[1].reshape_as(k);
                    } else if (!official_bf16 && q_norm.defined() && k_norm.defined() &&
                               q.scalar_type() == mfq_tensor_backend::kFloat16 &&
                               k.scalar_type() == mfq_tensor_backend::kFloat16) {
                        auto normalized = profiler.measure("full.qk_norm", [&]() {
                            return rms_norm_pair_f16_f32_offset_cuda(
                                q, k, q_norm, k_norm, rms_norm_eps, norm_weight_offset);
                        });
                        q = normalized[0].reshape_as(q);
                        k = normalized[1].reshape_as(k);
                    } else {
                        if (q_norm.defined())
                            q = profiler.measure("full.q_norm", [&]() {
                                return official_bf16
                                           ? qwen_rms_norm_bf16(execution.config.minicpm_fused_bf16_rmsnorm,
                                                                q.reshape({-1, hd}), q_norm,
                                                                rms_norm_eps, norm_weight_offset)
                                                 .reshape_as(q)
                                           : qwen_rms_norm(q.reshape({-1, hd}).to(
                                                               mfq_tensor_backend::kFloat32),
                                                           q_norm, rms_norm_eps, norm_weight_offset)
                                                 .reshape_as(q);
                            });
                        if (k_norm.defined())
                            k = profiler.measure("full.k_norm", [&]() {
                                return official_bf16
                                           ? qwen_rms_norm_bf16(execution.config.minicpm_fused_bf16_rmsnorm,
                                                                k.reshape({-1, hd}), k_norm,
                                                                rms_norm_eps, norm_weight_offset)
                                                 .reshape_as(k)
                                           : qwen_rms_norm(k.reshape({-1, hd}).to(
                                                               mfq_tensor_backend::kFloat32),
                                                           k_norm, rms_norm_eps, norm_weight_offset)
                                                 .reshape_as(k);
                            });
                    }
                    if (v_norm.defined())
                        v = profiler.measure("full.v_norm", [&]() {
                            return official_bf16
                                       ? qwen_rms_norm_bf16(execution.config.minicpm_fused_bf16_rmsnorm, v.reshape({-1, hd}),
                                                            v_norm, rms_norm_eps,
                                                            norm_weight_offset)
                                             .reshape_as(v)
                                       : qwen_rms_norm(
                                             v.reshape({-1, hd}).to(mfq_tensor_backend::kFloat32),
                                             v_norm, rms_norm_eps, norm_weight_offset)
                                             .reshape_as(v);
                        });
                },
                [&](auto &heads) {
                    auto &q = heads.query;
                    auto &k = heads.key;
                    auto &v = heads.value;
                    const bool fused_rope_kv =
                        apply_rotary && official_bf16 && x.is_cuda() &&
                        write_positions.dim() == 1 &&
                        pos.dim() == 1 && T == 1 && !cache.paged() && !cache.ring &&
                        active_rope.rotary_dim == 128 && active_rope.sections.numel() == 0 &&
                        nh == 32 && nkh == 8 && hd == 128 &&
                        cache.dtype == mfq_tensor_backend::kBFloat16 &&
                        execution.config.minicpm_fused_rope_kv;
                    if (!fused_rope_kv)
                        return false;

                    q = profiler.measure("full.rope_kv_write", [&]() {
                        return minicpm_bf16_rope_cache_write_cuda(
                            q.contiguous(), k.contiguous(), v.contiguous(),
                            pos.contiguous().to(x.device(), mfq_tensor_backend::kInt64),
                            write_positions, active_rope.cos, active_rope.sin, cache.physical_key, cache.physical_value,
                            active_rope.rotary_dim);
                    });
                    kv = {cache.physical_key.index({Slice(), Slice(), Slice(0, cache_pos + T), Slice()}),
                          cache.physical_value.index({Slice(), Slice(), Slice(0, cache_pos + T), Slice()})};
                    return true;
                },
                [&](auto &heads) {
                    auto &q = heads.query;
                    auto &k = heads.key;

                    q = profiler.measure("full.q_rope", [&]() {
                        return official_bf16 ? active_rope.apply_bf16(execution.config.minicpm_fused_bf16_rope, q, pos)
                                             : active_rope.apply(q, pos, grid_mrope_positions);
                    });
                    k = profiler.measure("full.k_rope", [&]() {
                        return official_bf16 ? active_rope.apply_bf16(execution.config.minicpm_fused_bf16_rope, k, pos)
                                             : active_rope.apply(k, pos, grid_mrope_positions);
                    });
                },
                [&](auto &heads) {
                    auto &k = heads.key;
                    auto &v = heads.value;
                    kv = profiler.measure("full.kv_write", [&]() {
                        return append_kv(cache, execution.config.kv_cache_write_aten, k, v, write_positions, cache_pos,
                                            cache_pos + T, !seq_len.has_value());
                    });
                },
                [&](auto &heads) {
                    trace_qwen_stage("qwen.q_rope", heads.query, 2);
                    trace_qwen_stage("qwen.k_rope", heads.key, 2);
                    cache.visible_key = kv.first;
                    cache.visible_value = kv.second;
                    auto result = mfq::cuda::dispatch_attention(
                        mfq::cuda::AttentionInput{
                            heads.query,
                            heads.key,
                            heads.value,
                            pos,
                            write_positions,
                            seq_len,
                            attention_mask,
                            cache_pos,
                            planned_kv_length,
                            decode_attention_parts},
                        cache,
                        mfq::cuda::AttentionConfig{
                            .official_bf16 = official_bf16,
                            .sliding = sliding,
                            .window = attention_window,
                            .scale = attention_scale,
                            .bf16_flash128 = execution.config.minicpm_bf16_flash128,
                            .flash128_casts = execution.config.minicpm_flash128_specialized_casts,
                            .bf16_gqa_decode = execution.config.minicpm_bf16_gqa_decode,
                            .decode_aten = execution.config.attention_decode_aten,
                            .mma_prefill = execution.config.mma_attention,
                            .mma_decode = execution.config.mma_attention_decode,
                            .split_k = execution.config.attention_decode_split_k},
                        scratch, profiler);
                    attention_token_major = result.token_major;
                    return result.value;
                },
                [&](mfq_tensor_backend::Tensor a, std::optional<mfq_tensor_backend::Tensor> gate) {
                    auto q_gate = gate.value_or(mfq_tensor_backend::Tensor{});
                    mfq_tensor_backend::Tensor oo;
                    if (q_gate.defined()) {
                        trace_qwen_stage("qwen.attention", a, attention_token_major ? 1 : 2);
                        auto af = profiler.measure("full.attn_out_view", [&]() {
                            return attention_token_major
                                       ? a.reshape({B, T, attn_width})
                                       : a.transpose(1, 2).contiguous().reshape({B, T, attn_width});
                        });
                        auto gf = profiler.measure("full.q_gate_view", [&]() {
                            return q_gate.contiguous().reshape({B, T, attn_width});
                        });
                        oo = profiler.measure("full.o_proj_gate", [&]() {
                            return o.forward_input_mul(execution, af, gf, InputGateMode::Sigmoid);
                        });
                    } else {
                        auto af = profiler.measure("full.attn_out_view", [&]() {
                            return attention_token_major
                                       ? a.reshape({B, T, attn_width})
                                       : a.transpose(1, 2).contiguous().reshape({B, T, attn_width});
                        });
                        oo = profiler.measure("full.o_proj", [&]() {
                            return official_bf16 ? o.forward_bf16_output(execution, af)
                                                 : o.forward(execution, af);
                        });
                    }
                    trace_qwen_stage("qwen.o_projection", oo);
                    if (official_bf16) {
                        oo = oo.to(mfq_tensor_backend::kBFloat16).contiguous();
                    }
                    return oo;
                },
                apply_rotary);
        },
        [&](mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor oo) {
            return std::array<mfq_tensor_backend::Tensor, 2>{std::move(residual), std::move(oo)};
        });
}

} // namespace mfq::cuda
