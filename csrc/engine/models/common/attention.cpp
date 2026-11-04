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

#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
using ::tyloi::indexing::Slice;
#endif

namespace tyloi::cuda {

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
std::array<tyloi::DenseTensor, 2> FullAttention::forward(
        tyloi::DenseTensor x, const AttentionStep& step, tyloi::state::KvView cache,
        const tyloi::ops::RopeTable& rope, tyloi::ops::Trace* trace) const {
    using DenseTensor = tyloi::DenseTensor;
    const auto profile_device = x.device();
    auto measure = [&](const char* name, auto&& operation) {
        return tyloi::ops::measure(name, profile_device, std::forward<decltype(operation)>(operation));
    };
#else
std::array<::tyloi::DenseTensor, 2> FullAttention::forward(
        const Projector& project, CudaProfiler& profiler, const AttentionOptions& options,
        ::tyloi::DenseTensor x,
        const AttentionStep& step, AttentionKvView cache,
        AttentionScratch& scratch, const RopeCache& rope, Trace* trace) const {
    using DenseTensor = ::tyloi::DenseTensor;
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
    std::int64_t B = x.size(0), T = x.size(1), H = x.size(2);
    auto trace_qwen_stage = [&](const char *name, const DenseTensor &value,
                                int token_axis = 1) {
        if (!attention_output_gate || trace == nullptr)
            return;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        auto ordered = token_axis == 1 ? value
            : tyloi::ops::contiguous(tyloi::ops::transpose(value, 1, token_axis));
        trace->emplace_back(name, tyloi::ops::clone(tyloi::ops::contiguous(tyloi::ops::cast(
            tyloi::ops::reshape(ordered, std::array{B, T, std::int64_t{-1}}), tyloi::ScalarType::float32))));
#else
        auto ordered = token_axis == 1 ? value : value.transpose(1, token_axis).contiguous();
        trace->emplace_back(name, ordered.reshape({B, T, -1})
                                     .to(::tyloi::kFloat32).contiguous().clone());
#endif
    };
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    if (official_bf16 && x.scalar_type() != tyloi::ScalarType::bfloat16) {
        x = tyloi::ops::contiguous(tyloi::ops::cast(x, tyloi::ScalarType::bfloat16));
#else
    if (official_bf16 && x.scalar_type() != ::tyloi::kBFloat16) {
        x = x.to(::tyloi::kBFloat16).contiguous();
#endif
    }
    const std::int64_t nh = attention_heads;
    const std::int64_t nkh = kv_heads;
    const std::int64_t hd = attention_head_dim;
    const std::int64_t attn_width = nh * hd;
    const auto& active_rope = attention_rope.cos.defined() ? attention_rope : rope;
    const bool apply_rotary = !no_rope;
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
    if (x.is_cuda())
        scratch.split(B * nh, AttentionScratch::kMaxParts, hd, x.device());
#endif
    DenseTensor physical_key(cache.physical_key), physical_value(cache.physical_value);
    auto residual = x;
    DenseTensor write_positions, q, k, v, q_gate;
    bool grid_mrope_positions;
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
    bool fused_qk_rope_kv;
#endif
    std::pair<DenseTensor, DenseTensor> kv;
    {
        auto xn = measure("full.attn_norm", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            return tyloi::ops::rms_norm(
                x, attn_norm, rms_norm_eps, norm_weight_offset,
                official_bf16 ? tyloi::ScalarType::bfloat16 : tyloi::ScalarType::float32);
#else
            return official_bf16
                       ? qwen_rms_norm_bf16(options.fused_norm, x.reshape({B * T, H}), attn_norm,
                                            rms_norm_eps, norm_weight_offset)
                             .reshape({B, T, H})
                       : qwen_rms_norm(x.reshape({B * T, H}).to(::tyloi::kFloat32),
                                       attn_norm, rms_norm_eps, norm_weight_offset)
                             .reshape({B, T, H});
#endif
        });
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        write_positions = tyloi::ops::contiguous(tyloi::ops::cast(
            cache_positions.has_value() ? *cache_positions : pos, tyloi::ScalarType::int64, x.device()));
#else
        write_positions =
            cache_positions.has_value()
                ? cache_positions.value()
                      .to(x.device(), ::tyloi::kInt64)
                      .contiguous()
                : pos.to(x.device(), ::tyloi::kInt64).contiguous();
#endif
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
            cache.dtype == ::tyloi::kBFloat16 &&
            options.fused_qk_rope_kv;
#endif
        trace_qwen_stage("qwen.attn_norm", xn);
        auto parts = measure("full.qkv", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            std::vector<tyloi::DenseTensor> result;
            if (!split_q_kv_projections) {
                return tyloi::ops::project_group(xn, *qkv);
            } else {
                result.emplace_back(tyloi::ops::project(xn, *q_projection));
                result.emplace_back(tyloi::ops::project(xn, *k_projection));
                result.emplace_back(tyloi::ops::project(xn, *v_projection));
            }
            return result;
#else
            if (!split_q_kv_projections) {
                return project.run(qkv, xn);
            }
            std::vector<::tyloi::DenseTensor> result;
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
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            q_full = tyloi::ops::contiguous(tyloi::ops::cast(q_full, tyloi::ScalarType::bfloat16));
            k_full = tyloi::ops::contiguous(tyloi::ops::cast(k_full, tyloi::ScalarType::bfloat16));
            v_full = tyloi::ops::contiguous(tyloi::ops::cast(v_full, tyloi::ScalarType::bfloat16));
#else
            q_full = q_full.to(::tyloi::kBFloat16).contiguous();
            k_full = k_full.to(::tyloi::kBFloat16).contiguous();
            v_full = v_full.to(::tyloi::kBFloat16).contiguous();
#endif
        }
        DenseTensor q_raw;
        measure("full.qkv_view", [&]() {
            if (attention_output_gate) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                auto qp = tyloi::ops::reshape(q_full, std::array{B, T, nh, hd * 2});
                auto chunks = tyloi::ops::chunk(qp, 2, -1);
#else
                auto qp = q_full.reshape({B, T, nh, hd * 2});
                auto chunks = qp.chunk(2, -1);
#endif
                q_raw = chunks[0];
                q_gate = chunks[1];
            } else {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                q_raw = tyloi::ops::reshape(q_full, std::array{B, T, nh, hd});
#else
                q_raw = q_full.reshape({B, T, nh, hd});
#endif
            }
            return q_raw;
        });
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        q = measure("full.q_view", [&]() { return tyloi::ops::contiguous(tyloi::ops::transpose(q_raw, 1, 2)); });
        k = measure("full.k_view", [&]() {
            return tyloi::ops::contiguous(tyloi::ops::transpose(
                tyloi::ops::reshape(k_full, std::array{B, T, nkh, hd}), 1, 2));
        });
        v = measure("full.v_view", [&]() {
            return tyloi::ops::contiguous(tyloi::ops::transpose(
                tyloi::ops::reshape(v_full, std::array{B, T, nkh, hd}), 1, 2));
        });
#else
        q = measure("full.q_view", [&]() { return q_raw.transpose(1, 2).contiguous(); });
        k = measure("full.k_view", [&]() {
            return k_full.reshape({B, T, nkh, hd}).transpose(1, 2).contiguous();
        });
        v = measure("full.v_view", [&]() {
            return v_full.reshape({B, T, nkh, hd}).transpose(1, 2).contiguous();
        });
#endif
    }
    bool prepared = false;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    if (apply_rotary && official_bf16 && !v_norm.defined()) {
        auto result = tyloi::ops::norm_rope_kv(
            tyloi::ops::AttentionInput{q, k, v, pos, write_positions}, cache,
            active_rope, q_norm, k_norm, rms_norm_eps, norm_weight_offset);
        if (result) {
            q = *result;
            prepared = true;
        }
    }
#else
    if (fused_qk_rope_kv) {
        q = measure("full.qk_norm_rope_kv_write", [&]() {
            return minicpm_qk_norm_rope_cache_write_bf16_cuda(
                q.contiguous(), k.contiguous(), v.contiguous(), q_norm, k_norm,
                pos.contiguous().to(x.device(), ::tyloi::kInt64),
                write_positions, active_rope.cos, active_rope.sin, physical_key, physical_value,
                rms_norm_eps, norm_weight_offset);
        });
        prepared = true;
    }
#endif
    if (prepared) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        kv = {tyloi::ops::slice(physical_key, 2, 0, cache_pos + T),
              tyloi::ops::slice(physical_value, 2, 0, cache_pos + T)};
#else
        kv = {physical_key.index({Slice(), Slice(), Slice(0, cache_pos + T), Slice()}),
              physical_value.index({Slice(), Slice(), Slice(0, cache_pos + T), Slice()})};
#endif
    } else {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        {
            auto normalized = tyloi::ops::qkv_norm({q, k, v}, {q_norm, k_norm, v_norm},
                rms_norm_eps, norm_weight_offset,
                official_bf16 ? tyloi::ScalarType::bfloat16 : tyloi::ScalarType::float32);
            q = normalized[0];
            k = normalized[1];
            v = normalized[2];
        }
#else
        const bool fused_bf16_norm =
            official_bf16 && options.fused_norm;
        if (fused_bf16_norm && q_norm.defined() && k_norm.defined() &&
            q.scalar_type() == ::tyloi::kBFloat16 &&
            k.scalar_type() == ::tyloi::kBFloat16) {
            auto normalized = measure("full.qk_norm", [&]() {
                return qwen_rms_norm_pair_bf16_cuda(q, k, q_norm, k_norm, rms_norm_eps,
                                                    norm_weight_offset);
            });
            q = normalized[0].reshape_as(q);
            k = normalized[1].reshape_as(k);
        } else if (!official_bf16 && q_norm.defined() && k_norm.defined() &&
                   q.scalar_type() == ::tyloi::kFloat16 &&
                   k.scalar_type() == ::tyloi::kFloat16) {
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
                                                   ::tyloi::kFloat32),
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
                                                   ::tyloi::kFloat32),
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
                                 v.reshape({-1, hd}).to(::tyloi::kFloat32),
                                 v_norm, rms_norm_eps, norm_weight_offset)
                                 .reshape_as(v);
            });
#endif
        bool rotary_cached = false;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if (apply_rotary && official_bf16) {
            auto result = tyloi::ops::rope_kv(
                tyloi::ops::AttentionInput{q, k, v, pos, write_positions}, cache, active_rope);
            if (result) {
                q = *result;
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
            cache.dtype == ::tyloi::kBFloat16 &&
            options.fused_rope_kv;
        if (fused_rope_kv) {
            q = measure("full.rope_kv_write", [&]() {
                return minicpm_bf16_rope_cache_write_cuda(
                    q.contiguous(), k.contiguous(), v.contiguous(),
                    pos.contiguous().to(x.device(), ::tyloi::kInt64),
                    write_positions, active_rope.cos, active_rope.sin, physical_key, physical_value,
                    active_rope.rotary_dim);
            });
            rotary_cached = true;
        }
#endif
        if (rotary_cached) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            kv = {tyloi::ops::slice(physical_key, 2, 0, cache_pos + T),
                  tyloi::ops::slice(physical_value, 2, 0, cache_pos + T)};
#else
            kv = {physical_key.index({Slice(), Slice(), Slice(0, cache_pos + T), Slice()}),
                  physical_value.index({Slice(), Slice(), Slice(0, cache_pos + T), Slice()})};
#endif
        } else {
            if (apply_rotary) {
                q = measure("full.q_rope", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                    return tyloi::ops::rope(q, pos, active_rope, official_bf16, grid_mrope_positions);
#else
                    return official_bf16 ? active_rope.apply_bf16(options.fused_rope, q, pos)
                                         : active_rope.apply(q, pos, grid_mrope_positions);
#endif
                });
                k = measure("full.k_rope", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                    return tyloi::ops::rope(k, pos, active_rope, official_bf16, grid_mrope_positions);
#else
                    return official_bf16 ? active_rope.apply_bf16(options.fused_rope, k, pos)
                                         : active_rope.apply(k, pos, grid_mrope_positions);
#endif
                });
            }
            kv = measure("full.kv_write", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                return tyloi::ops::append_kv(cache, k, v, write_positions, cache_pos,
                                           !seq_len.has_value());
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
    auto result = tyloi::ops::attention(
        tyloi::ops::AttentionInput{q, k, v, pos,
            write_positions, seq_len, attention_mask, cache_pos, planned_kv_length,
            decode_attention_parts}, cache,
        tyloi::ops::AttentionConfig{official_bf16, sliding, attention_window,
                                  attention_scale});
#else
    auto result = tyloi::cuda::dispatch_attention(
        tyloi::cuda::AttentionInput{
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
        tyloi::cuda::AttentionConfig{
            .official_bf16 = official_bf16,
            .sliding = sliding,
            .window = attention_window,
            .scale = attention_scale,
            .kernels = options.kernels},
        scratch, profiler);
#endif
    const bool attention_token_major = result.token_major;
    auto a = result.value;
    DenseTensor oo;
    if (q_gate.defined()) {
        trace_qwen_stage("qwen.attention", a, attention_token_major ? 1 : 2);
        auto af = measure("full.attn_out_view", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            return tyloi::ops::reshape(attention_token_major ? a
                : tyloi::ops::contiguous(tyloi::ops::transpose(a, 1, 2)), std::array{B, T, attn_width});
#else
            return attention_token_major
                       ? a.reshape({B, T, attn_width})
                       : a.transpose(1, 2).contiguous().reshape({B, T, attn_width});
#endif
        });
        auto gf = measure("full.q_gate_view", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            return tyloi::ops::reshape(tyloi::ops::contiguous(q_gate), std::array{B, T, attn_width});
#else
            return q_gate.contiguous().reshape({B, T, attn_width});
#endif
        });
        oo = measure("full.o_proj_gate", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            return tyloi::ops::sigmoid_project(af, gf, *o);
#else
            return project.gated(o, af, gf, InputGateMode::Sigmoid);
#endif
        });
    } else {
        auto af = measure("full.attn_out_view", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            return tyloi::ops::reshape(attention_token_major ? a
                : tyloi::ops::contiguous(tyloi::ops::transpose(a, 1, 2)), std::array{B, T, attn_width});
#else
            return attention_token_major
                       ? a.reshape({B, T, attn_width})
                       : a.transpose(1, 2).contiguous().reshape({B, T, attn_width});
#endif
        });
        oo = measure("full.o_proj", [&]() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            auto output = tyloi::ops::project(af, *o);
            return official_bf16
                ? tyloi::ops::contiguous(tyloi::ops::cast(output, tyloi::ScalarType::bfloat16)) : output;
#else
            auto output = project.run(o, af);
            return official_bf16 ? output.to(::tyloi::kBFloat16).contiguous()
                                 : output;
#endif
        });
    }
    trace_qwen_stage("qwen.o_projection", oo);
    if (official_bf16) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        oo = tyloi::ops::contiguous(tyloi::ops::cast(oo, tyloi::ScalarType::bfloat16));
#else
        oo = oo.to(::tyloi::kBFloat16).contiguous();
#endif
    }
    return {std::move(residual), std::move(oo)};
}

} // namespace tyloi::cuda
