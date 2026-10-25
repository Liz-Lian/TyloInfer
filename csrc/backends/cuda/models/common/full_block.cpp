#include "models/common/full_block.h"
#include "backends/cuda/engine/executor/options.h"
#include "engine/models/common/gated_mlp.h"
#include "engine/models/common/transformer_layer.h"
#include <array>

#include "mfq_cuda_activation_ops.h"
#include "mfq_cuda_norm_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

mfq_tensor_backend::Tensor Block::forward_context(CudaExecutionContext &execution,
                                                  mfq_tensor_backend::Tensor x,
                                                  const Context &context, const RopeCache &rope) {
    TYLOINFER_RUNTIME_CHECK(context.confirmed_prefix == 0 || supports_speculation(),
                      "block does not support speculative verification");
    return forward(execution, std::move(x), context.positions, context.cache_position,
                   context.sequence_lengths, rope, context.cache_positions, context.attention_mask);
}

void FullBlock::preallocate_kv() {
    if (!mfq::cuda::runtime_options::kv_vram_preallocated() || cpu_offloaded || cache.defined()) return;
    MfqCudaGuard guard(cuda_device);
    cache = KVCache(1, kv_heads, sliding ? attention_window : max_position_embeddings,
        attention_head_dim, sliding, mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, cuda_device),
        official_bf16 ? mfq_tensor_backend::kBFloat16 : mfq_tensor_backend::kFloat16, kv_quantization);
}

void FullBlock::reset(int64_t B) {
    if (cache.defined() && cache.batch_size() == B)
        return;
    cache = KVCache();
}

mfq_tensor_backend::Tensor
FullBlock::forward(CudaExecutionContext &execution, mfq_tensor_backend::Tensor x,
                   mfq_tensor_backend::Tensor pos, int64_t cache_pos,
                   const MfqOptional<mfq_tensor_backend::Tensor> &seq_len, const RopeCache &rope,
                   const MfqOptional<mfq_tensor_backend::Tensor> &cache_positions,
                   const MfqOptional<mfq_tensor_backend::Tensor> &attention_mask) {
    return forward_impl(execution, std::move(x), std::move(pos), cache_pos, seq_len, rope,
                        cache_positions, attention_mask, 0, 0);
}

mfq_tensor_backend::Tensor FullBlock::forward_context(CudaExecutionContext &execution,
                                                      mfq_tensor_backend::Tensor x,
                                                      const Context &context,
                                                      const RopeCache &rope) {
    TYLOINFER_RUNTIME_CHECK(context.confirmed_prefix == 0 || supports_speculation(),
                      "block does not support speculative verification");
    return forward_impl(execution, std::move(x), context.positions, context.cache_position,
                        context.sequence_lengths, rope, context.cache_positions,
                        context.attention_mask, context.planned_kv_length,
                        context.decode_attention_parts);
}

mfq_tensor_backend::Tensor FullBlock::forward_impl(
    CudaExecutionContext &execution, mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor pos,
    int64_t cache_pos, const MfqOptional<mfq_tensor_backend::Tensor> &seq_len,
    const RopeCache &rope, const MfqOptional<mfq_tensor_backend::Tensor> &cache_positions,
    const MfqOptional<mfq_tensor_backend::Tensor> &attention_mask, int64_t planned_kv_length,
    int64_t decode_attention_parts) {
    const int64_t B = x.size(0), T = x.size(1), H = x.size(2);
    const auto kl_capacity = execution.kl_mmq.kv_cache_capacity;
    const int64_t capacity = sliding ? attention_window
        : (kl_capacity > 0 ? std::max<int64_t>(cache_pos + T, kl_capacity)
                           : max_position_embeddings);
    if (!cache.defined()) {
        cache = KVCache(B, kv_heads, capacity, attention_head_dim, sliding, x.device(),
            official_bf16 ? mfq_tensor_backend::kBFloat16 : mfq_tensor_backend::kFloat16,
            kv_quantization);
    }
    cache.quantization = kv_quantization;
    const mfq::cuda::AttentionStep step{
        std::move(pos), seq_len, cache_positions, attention_mask,
        cache_pos, planned_kv_length, decode_attention_parts};
    auto attention = FullAttention::forward(
        execution, std::move(x), step, cache.view(), execution.attention_scratch[this], rope);
    return forward_ffn(execution, std::move(attention[0]), std::move(attention[1]), B, T, H);
}

mfq_tensor_backend::Tensor FullBlock::forward_ffn(CudaExecutionContext &execution,
                                                  mfq_tensor_backend::Tensor residual,
                                                  mfq_tensor_backend::Tensor attention_output,
                                                  int64_t B, int64_t T, int64_t H) {
    auto &profiler = execution.profiler;
    auto oo = std::move(attention_output);
    auto trace_qwen_stage = [&](const char *name, const mfq_tensor_backend::Tensor &value) {
        if (!attention_output_gate || execution.gemma_stage_trace == nullptr ||
            layer != execution.gemma_trace_layer)
            return;
        trace_gemma_stage(execution, layer, name, value.reshape({B, T, -1}));
    };
    return mfq::models::residual_ffn(
        std::move(residual), std::move(oo),
        [&](mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor oo) {
            auto attn_pair = profiler.measure("full.attn_residual_ffn_norm", [&]() {
                auto rr = residual.reshape({-1, H});
                auto oo2 = oo.reshape({-1, H});
                if (!rr.is_cuda()) {
                    auto summed =
                        (rr.to(mfq_tensor_backend::kFloat32) + oo2.to(mfq_tensor_backend::kFloat32))
                            .to(rr.scalar_type())
                            .contiguous();
                    auto normalized =
                        official_bf16 ? qwen_rms_norm_bf16(execution.config.minicpm_fused_bf16_rmsnorm, summed, ffn_norm,
                                                           rms_norm_eps, norm_weight_offset)
                                      : qwen_rms_norm(summed.to(mfq_tensor_backend::kFloat32),
                                                      ffn_norm, rms_norm_eps, norm_weight_offset);
                    return std::vector<mfq_tensor_backend::Tensor>{summed, normalized};
                }
                if (oo2.scalar_type() != rr.scalar_type()) {
                    oo2 = oo2.to(rr.scalar_type()).contiguous();
                }
                if (execution.config.diagnostic_fp32_residual) {
                    return acc_rms_norm_cuda(rr.to(mfq_tensor_backend::kFloat32),
                                             oo2.to(mfq_tensor_backend::kFloat32), ffn_norm,
                                             rms_norm_eps, norm_weight_offset);
                }
                if (rr.scalar_type() == mfq_tensor_backend::kFloat16 &&
                    oo2.scalar_type() == mfq_tensor_backend::kFloat16) {
                    return acc_rms_norm_f16_cuda(rr, oo2, ffn_norm, rms_norm_eps,
                                                 norm_weight_offset);
                }
                if (rr.scalar_type() == mfq_tensor_backend::kBFloat16 &&
                    oo2.scalar_type() == mfq_tensor_backend::kBFloat16) {
                    if (official_bf16) {
                        return acc_rms_norm_bf16_cuda(rr, oo2, ffn_norm, rms_norm_eps,
                                                      norm_weight_offset);
                    }
                    auto sum = (rr + oo2).contiguous();
                    auto norm = qwen_rms_norm(sum.to(mfq_tensor_backend::kFloat32), ffn_norm,
                                              rms_norm_eps, norm_weight_offset)
                                    .to(mfq_tensor_backend::kBFloat16)
                                    .contiguous();
                    return std::vector<mfq_tensor_backend::Tensor>{sum, norm};
                }
                return acc_rms_norm_cuda(rr, oo2, ffn_norm, rms_norm_eps, norm_weight_offset);
            });
            return std::array<mfq_tensor_backend::Tensor, 2>{attn_pair[0].reshape({B, T, H}),
                                                             attn_pair[1].reshape({B, T, H})};
        },
        [&](mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor xn) {
            trace_qwen_stage("qwen.ffn_norm", xn);
            return mfq::models::feed_forward(
                std::move(residual), std::move(xn),
                [&](mfq_tensor_backend::Tensor xn, mfq_tensor_backend::Tensor residual)
                    -> std::optional<mfq_tensor_backend::Tensor> {
                    if (!official_bf16) {
                        auto ffn_input = xn.reshape({B * T, H});
                        auto residual_flat = residual.reshape({B * T, H});
                        if (ffn.can_forward_fused_residual(execution.config, ffn_input,
                                                           residual_flat)) {
                            return profiler.measure("full.ffn_down_residual", [&]() {
                                return ffn
                                    .forward_fused_residual(profiler, execution.config, ffn_input,
                                                            residual_flat)
                                    .reshape({B, T, H});
                            });
                        }
                    }
                    return {};
                },
                [&](mfq_tensor_backend::Tensor xn) {
                    mfq_tensor_backend::Tensor ff;
                    if (official_bf16) {
                        auto ffn_input = xn.reshape({B * T, H});
                        using Tensor = mfq_tensor_backend::Tensor;
                        using Activation = mfq::models::GatedActivation;
                        ff = mfq::models::gated_mlp(
                                 ffn_input, false, 0.0,
                                 [](const Tensor &, Activation, double) {
                                     return std::optional<Tensor>{};
                                 },
                                 [](const Tensor &, Activation, double) {
                                     return std::optional<Tensor>{};
                                 },
                                 [&](Tensor ffn_input) {
                                     return profiler.measure("full.minicpmo45_ffn_gate_up", [&]() {
                                         return ffn.gate_up.forward(execution, ffn_input);
                                     });
                                 },
                                 [&](Tensor gate, Tensor up, Activation, double) {
                                     gate = gate.to(mfq_tensor_backend::kBFloat16).contiguous();
                                     up = up.to(mfq_tensor_backend::kBFloat16).contiguous();
                                     return profiler.measure("full.minicpmo45_ffn_swiglu", [&]() {
                                         if (execution.config.minicpm_bf16_swiglu_fusion) {
                                             return silu_mul_cuda(gate, up);
                                         }
                                         return (mfq_tensor_backend::silu(gate) * up).contiguous();
                                     });
                                 },
                                 [&](Tensor activation) {
                                     return profiler.measure("full.minicpmo45_ffn_down", [&] {
                                         return ffn.down.forward_bf16_output(execution, activation);
                                     });
                                 },
                                 [](const Tensor &, const Tensor &, Activation, double) {
                                     return std::optional<Tensor>{};
                                 })
                                 .reshape({B, T, H})
                                 .to(mfq_tensor_backend::kBFloat16)
                                 .contiguous();
                    } else {
                        ff = ffn.forward(execution, xn.reshape({B * T, H})).reshape({B, T, H});
                    }

                    return ff;
                },
                [&](mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor ff) {
                    trace_qwen_stage("qwen.ffn_output", ff);
                    auto output = profiler.measure("full.ffn_residual", [&]() {
                        auto rr = residual.reshape({-1, H});
                        auto ff2 = ff.reshape({-1, H});
                        if (!rr.is_cuda()) {
                            return (rr.to(mfq_tensor_backend::kFloat32) +
                                    ff2.to(mfq_tensor_backend::kFloat32))
                                .to(rr.scalar_type())
                                .reshape({B, T, H})
                                .contiguous();
                        }
                        if (execution.config.diagnostic_fp32_residual) {
                            rr = rr.to(mfq_tensor_backend::kFloat32);
                            ff2 = ff2.to(mfq_tensor_backend::kFloat32);
                        } else if (ff2.scalar_type() != rr.scalar_type()) {
                            ff2 = ff2.to(rr.scalar_type()).contiguous();
                        }
                        if (rr.scalar_type() == mfq_tensor_backend::kBFloat16 &&
                            ff2.scalar_type() == mfq_tensor_backend::kBFloat16) {
                            if (execution.config.minicpm_bf16_residual_acc) {
                                return acc_cuda(rr, ff2).reshape({B, T, H});
                            }
                            return (rr + ff2).contiguous().reshape({B, T, H});
                        }
                        return acc_cuda(rr, ff2).reshape({B, T, H});
                    });
                    return output;
                });
        });
}
