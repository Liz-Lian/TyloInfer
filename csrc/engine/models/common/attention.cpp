#include "engine/models/common/attention.h"
#include "engine/ops.h"
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/backends/cuda/ops/attention/kv.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/kernels/mfq_cuda_activation_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_attention_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#endif

#include <stdexcept>
#include <utility>
#include <vector>

using mfq_tensor_backend::indexing::Slice;

namespace mfq::cuda {

std::array<mfq_tensor_backend::Tensor, 2> FullAttention::forward(
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        mfq_tensor_backend::Tensor x, const AttentionStep& step, AttentionKvView cache,
        const RopeCache& rope, const mfq::ops::ProjectionOptions& projection, Trace* trace) const {
    using mfq::ops::measure;
#else
        const Projector& project, CudaProfiler& profiler, const AttentionOptions& options,
        mfq_tensor_backend::Tensor x,
        const AttentionStep& step, AttentionKvView cache,
        AttentionScratch& scratch, const RopeCache& rope, Trace* trace) const {
    auto measure = [&](const char* name, auto&& operation) {
        return profiler.measure(name, std::forward<decltype(operation)>(operation));
    };
#endif
    const auto& pos = step.positions;
    const auto& seq_len = step.sequence_lengths;
    const auto& cache_positions = step.cache_positions;
    const auto& attention_mask = step.attention_mask;
    const auto cache_pos = step.cache_position;
    const auto planned_kv_length = step.planned_kv_length;
    const auto decode_attention_parts = step.decode_attention_parts;
    int64_t B = x.size(0), T = x.size(1), H = x.size(2);
    auto trace_qwen_stage = [&](const char *name, const mfq_tensor_backend::Tensor &value,
                                int token_axis = 1) {
        if (!attention_output_gate || trace == nullptr)
            return;
        auto ordered = token_axis == 1 ? value : value.transpose(1, token_axis).contiguous();
        trace->emplace_back(name, ordered.reshape({B, T, -1})
                                     .to(mfq_tensor_backend::kFloat32).contiguous().clone());
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
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
    if (x.is_cuda())
        scratch.split(B * nh, kDecodeAttentionMaxParts, hd, x.device());
#endif
    mfq_tensor_backend::Tensor physical_key(cache.physical_key), physical_value(cache.physical_value);
    auto residual = x;
    mfq_tensor_backend::Tensor write_positions, q, k, v, q_gate;
    bool grid_mrope_positions;
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
    bool fused_qk_rope_kv;
#endif
    std::pair<mfq_tensor_backend::Tensor, mfq_tensor_backend::Tensor> kv;
    {
        auto xn = measure("full.attn_norm", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            return mfq_tensor_backend::Tensor(mfq::ops::rms_norm(
                x, attn_norm, rms_norm_eps, norm_weight_offset,
                official_bf16 ? mfq::ScalarType::bfloat16 : mfq::ScalarType::float32));
#else
            return official_bf16
                       ? qwen_rms_norm_bf16(options.fused_norm, x.reshape({B * T, H}), attn_norm,
                                            rms_norm_eps, norm_weight_offset)
                             .reshape({B, T, H})
                       : qwen_rms_norm(x.reshape({B * T, H}).to(mfq_tensor_backend::kFloat32),
                                       attn_norm, rms_norm_eps, norm_weight_offset)
                             .reshape({B, T, H});
#endif
        });
        write_positions =
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
        grid_mrope_positions =
            B == 1 && cache_positions.has_value() && pos.dim() == 2 && pos.size(0) == 3;
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
        fused_qk_rope_kv =
            apply_rotary && official_bf16 && x.is_cuda() &&
            write_positions.dim() == 1 && pos.dim() == 1 &&
            T == 1 && !cache.paged() && !cache.ring && !cache.quantization.enabled() && !v_norm.defined() &&
            q_norm.defined() && k_norm.defined() && active_rope.rotary_dim == 128 &&
            active_rope.sections.numel() == 0 && nh == 32 && nkh == 8 && hd == 128 &&
            cache.dtype == mfq_tensor_backend::kBFloat16 &&
            options.fused_qk_rope_kv;
#endif
        trace_qwen_stage("qwen.attn_norm", xn);
        auto parts = measure("full.qkv", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            std::vector<mfq_tensor_backend::Tensor> result;
            if (!split_q_kv_projections) {
                auto parts = mfq::ops::project(xn, qkv, projection);
                for (auto& part : parts) result.emplace_back(part);
            } else {
                result.emplace_back(mfq::ops::project(xn, q_projection, projection));
                result.emplace_back(mfq::ops::project(xn, k_projection, projection));
                result.emplace_back(mfq::ops::project(xn, v_projection, projection));
            }
            return result;
#else
            if (!split_q_kv_projections) {
                return project.run(qkv, xn);
            }
            std::vector<mfq_tensor_backend::Tensor> result;
            result.reserve(3);
            result.push_back(project.run(q_projection, xn));
            result.push_back(project.run(k_projection, xn));
            result.push_back(project.run(v_projection, xn));
            return result;
#endif
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
        mfq_tensor_backend::Tensor q_raw;
        measure("full.qkv_view", [&]() {
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
        q = measure("full.q_view", [&]() { return q_raw.transpose(1, 2).contiguous(); });
        k = measure("full.k_view", [&]() {
            return k_full.reshape({B, T, nkh, hd}).transpose(1, 2).contiguous();
        });
        v = measure("full.v_view", [&]() {
            return v_full.reshape({B, T, nkh, hd}).transpose(1, 2).contiguous();
        });
    }
    bool prepared = false;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    if (apply_rotary && official_bf16 && !v_norm.defined()) {
        auto result = mfq::ops::norm_rope_kv(
            mfq::ops::AttentionInput{q, k, v, pos, write_positions}, cache,
            active_rope.view(), q_norm, k_norm, rms_norm_eps, norm_weight_offset);
        if (result) {
            q = mfq_tensor_backend::Tensor(*result);
            prepared = true;
        }
    }
#else
    if (fused_qk_rope_kv) {
        q = measure("full.qk_norm_rope_kv_write", [&]() {
            return minicpm_qk_norm_rope_cache_write_bf16_cuda(
                q.contiguous(), k.contiguous(), v.contiguous(), q_norm, k_norm,
                pos.contiguous().to(x.device(), mfq_tensor_backend::kInt64),
                write_positions, active_rope.cos, active_rope.sin, physical_key, physical_value,
                rms_norm_eps, norm_weight_offset);
        });
        prepared = true;
    }
#endif
    if (prepared) {
        kv = {physical_key.index({Slice(), Slice(), Slice(0, cache_pos + T), Slice()}),
              physical_value.index({Slice(), Slice(), Slice(0, cache_pos + T), Slice()})};
    } else {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        {
            auto normalized = mfq::ops::qkv_norm({q, k, v}, {q_norm, k_norm, v_norm},
                rms_norm_eps, norm_weight_offset,
                official_bf16 ? mfq::ScalarType::bfloat16 : mfq::ScalarType::float32);
            q = mfq_tensor_backend::Tensor(normalized[0]);
            k = mfq_tensor_backend::Tensor(normalized[1]);
            v = mfq_tensor_backend::Tensor(normalized[2]);
        }
#else
        const bool fused_bf16_norm =
            official_bf16 && options.fused_norm;
        if (fused_bf16_norm && q_norm.defined() && k_norm.defined() &&
            q.scalar_type() == mfq_tensor_backend::kBFloat16 &&
            k.scalar_type() == mfq_tensor_backend::kBFloat16) {
            auto normalized = measure("full.qk_norm", [&]() {
                return qwen_rms_norm_pair_bf16_cuda(q, k, q_norm, k_norm, rms_norm_eps,
                                                    norm_weight_offset);
            });
            q = normalized[0].reshape_as(q);
            k = normalized[1].reshape_as(k);
        } else if (!official_bf16 && q_norm.defined() && k_norm.defined() &&
                   q.scalar_type() == mfq_tensor_backend::kFloat16 &&
                   k.scalar_type() == mfq_tensor_backend::kFloat16) {
            auto normalized = measure("full.qk_norm", [&]() {
                return rms_norm_pair_f16_f32_offset_cuda(
                    q, k, q_norm, k_norm, rms_norm_eps, norm_weight_offset);
            });
            q = normalized[0].reshape_as(q);
            k = normalized[1].reshape_as(k);
        } else {
            if (q_norm.defined())
                q = measure("full.q_norm", [&]() {
                    return official_bf16
                               ? qwen_rms_norm_bf16(options.fused_norm,
                                                    q.reshape({-1, hd}), q_norm,
                                                    rms_norm_eps, norm_weight_offset)
                                     .reshape_as(q)
                               : qwen_rms_norm(q.reshape({-1, hd}).to(
                                                   mfq_tensor_backend::kFloat32),
                                               q_norm, rms_norm_eps, norm_weight_offset)
                                     .reshape_as(q);
                });
            if (k_norm.defined())
                k = measure("full.k_norm", [&]() {
                    return official_bf16
                               ? qwen_rms_norm_bf16(options.fused_norm,
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
            v = measure("full.v_norm", [&]() {
                return official_bf16
                           ? qwen_rms_norm_bf16(options.fused_norm, v.reshape({-1, hd}),
                                                v_norm, rms_norm_eps,
                                                norm_weight_offset)
                                 .reshape_as(v)
                           : qwen_rms_norm(
                                 v.reshape({-1, hd}).to(mfq_tensor_backend::kFloat32),
                                 v_norm, rms_norm_eps, norm_weight_offset)
                                 .reshape_as(v);
            });
#endif
        bool rotary_cached = false;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if (apply_rotary && official_bf16) {
            auto result = mfq::ops::rope_kv(
                mfq::ops::AttentionInput{q, k, v, pos, write_positions}, cache, active_rope.view());
            if (result) {
                q = mfq_tensor_backend::Tensor(*result);
                rotary_cached = true;
            }
        }
#else
        const bool fused_rope_kv =
            apply_rotary && official_bf16 && x.is_cuda() &&
            write_positions.dim() == 1 &&
            pos.dim() == 1 && T == 1 && !cache.paged() && !cache.ring &&
            active_rope.rotary_dim == 128 && active_rope.sections.numel() == 0 &&
            nh == 32 && nkh == 8 && hd == 128 &&
            cache.dtype == mfq_tensor_backend::kBFloat16 &&
            options.fused_rope_kv;
        if (fused_rope_kv) {
            q = measure("full.rope_kv_write", [&]() {
                return minicpm_bf16_rope_cache_write_cuda(
                    q.contiguous(), k.contiguous(), v.contiguous(),
                    pos.contiguous().to(x.device(), mfq_tensor_backend::kInt64),
                    write_positions, active_rope.cos, active_rope.sin, physical_key, physical_value,
                    active_rope.rotary_dim);
            });
            rotary_cached = true;
        }
#endif
        if (rotary_cached) {
            kv = {physical_key.index({Slice(), Slice(), Slice(0, cache_pos + T), Slice()}),
                  physical_value.index({Slice(), Slice(), Slice(0, cache_pos + T), Slice()})};
        } else {
            if (apply_rotary) {
                q = measure("full.q_rope", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                    return mfq_tensor_backend::Tensor(mfq::ops::rope(
                        q, pos, active_rope.view(), official_bf16, grid_mrope_positions));
#else
                    return official_bf16 ? active_rope.apply_bf16(options.fused_rope, q, pos)
                                         : active_rope.apply(q, pos, grid_mrope_positions);
#endif
                });
                k = measure("full.k_rope", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                    return mfq_tensor_backend::Tensor(mfq::ops::rope(
                        k, pos, active_rope.view(), official_bf16, grid_mrope_positions));
#else
                    return official_bf16 ? active_rope.apply_bf16(options.fused_rope, k, pos)
                                         : active_rope.apply(k, pos, grid_mrope_positions);
#endif
                });
            }
            kv = measure("full.kv_write", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                auto result = mfq::ops::append_kv(cache, k, v, write_positions, cache_pos,
                                                  !seq_len.has_value());
                return std::pair{mfq_tensor_backend::Tensor(result.first),
                                 mfq_tensor_backend::Tensor(result.second)};
#else
                return append_kv(cache, options.write_aten, k, v, write_positions, cache_pos,
                                 cache_pos + T, !seq_len.has_value());
#endif
            });
        }
    }
    trace_qwen_stage("qwen.q_rope", q, 2);
    trace_qwen_stage("qwen.k_rope", k, 2);
    cache.visible_key = kv.first;
    cache.visible_value = kv.second;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    auto result = mfq::ops::attention(
        mfq::ops::AttentionInput{q, k, v, pos,
            write_positions, seq_len, attention_mask, cache_pos, planned_kv_length,
            decode_attention_parts}, cache,
        mfq::ops::AttentionConfig{official_bf16, sliding, attention_window,
                                  attention_scale});
#else
    auto result = mfq::cuda::dispatch_attention(
        mfq::cuda::AttentionInput{
            q,
            k,
            v,
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
            .kernels = options.kernels},
        scratch, profiler);
#endif
    const bool attention_token_major = result.token_major;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    auto a = mfq_tensor_backend::Tensor(result.value);
#else
    auto a = result.value;
#endif
    mfq_tensor_backend::Tensor oo;
    if (q_gate.defined()) {
        trace_qwen_stage("qwen.attention", a, attention_token_major ? 1 : 2);
        auto af = measure("full.attn_out_view", [&]() {
            return attention_token_major
                       ? a.reshape({B, T, attn_width})
                       : a.transpose(1, 2).contiguous().reshape({B, T, attn_width});
        });
        auto gf = measure("full.q_gate_view", [&]() {
            return q_gate.contiguous().reshape({B, T, attn_width});
        });
        oo = measure("full.o_proj_gate", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            return mfq_tensor_backend::Tensor(mfq::ops::sigmoid_project(af, gf, o, projection));
#else
            return project.gated(o, af, gf, InputGateMode::Sigmoid);
#endif
        });
    } else {
        auto af = measure("full.attn_out_view", [&]() {
            return attention_token_major
                       ? a.reshape({B, T, attn_width})
                       : a.transpose(1, 2).contiguous().reshape({B, T, attn_width});
        });
        oo = measure("full.o_proj", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            auto output = mfq_tensor_backend::Tensor(mfq::ops::project(af, o, projection));
#else
            auto output = project.run(o, af);
#endif
            return official_bf16 ? output.to(mfq_tensor_backend::kBFloat16).contiguous()
                                 : output;
        });
    }
    trace_qwen_stage("qwen.o_projection", oo);
    if (official_bf16) {
        oo = oo.to(mfq_tensor_backend::kBFloat16).contiguous();
    }
    return {std::move(residual), std::move(oo)};
}

} // namespace mfq::cuda
