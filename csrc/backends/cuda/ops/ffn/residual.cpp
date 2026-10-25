#include "residual.h"
#include "ops/norm/rms.h"
#include "engine/models/common/transformer_layer.h"
#include "mfq_cuda_activation_ops.h"
#include "mfq_cuda_norm_ops.h"

#include <array>
#include <optional>
#include <utility>
#include <vector>

namespace mfq::cuda {

mfq_tensor_backend::Tensor full_ffn(
    CudaExecutionContext& execution, const FFN& ffn,
    const mfq_tensor_backend::Tensor& ffn_norm, const FullAttention& attention,
    mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor attention_output) {
    const int64_t B = residual.size(0), T = residual.size(1), H = residual.size(2);
    auto &profiler = execution.profiler;
    auto oo = std::move(attention_output);
    auto trace_qwen_stage = [&](const char *name, const mfq_tensor_backend::Tensor &value) {
        if (!attention.attention_output_gate || execution.gemma_stage_trace == nullptr ||
            attention.layer != execution.gemma_trace_layer)
            return;
        trace_gemma_stage(execution, attention.layer, name, value.reshape({B, T, -1}));
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
                        attention.official_bf16 ? qwen_rms_norm_bf16(execution.config.minicpm_fused_bf16_rmsnorm, summed, ffn_norm,
                                                           attention.rms_norm_eps, attention.norm_weight_offset)
                                      : qwen_rms_norm(summed.to(mfq_tensor_backend::kFloat32),
                                                      ffn_norm, attention.rms_norm_eps, attention.norm_weight_offset);
                    return std::vector<mfq_tensor_backend::Tensor>{summed, normalized};
                }
                if (oo2.scalar_type() != rr.scalar_type()) {
                    oo2 = oo2.to(rr.scalar_type()).contiguous();
                }
                if (execution.config.diagnostic_fp32_residual) {
                    return acc_rms_norm_cuda(rr.to(mfq_tensor_backend::kFloat32),
                                             oo2.to(mfq_tensor_backend::kFloat32), ffn_norm,
                                             attention.rms_norm_eps, attention.norm_weight_offset);
                }
                if (rr.scalar_type() == mfq_tensor_backend::kFloat16 &&
                    oo2.scalar_type() == mfq_tensor_backend::kFloat16) {
                    return acc_rms_norm_f16_cuda(rr, oo2, ffn_norm, attention.rms_norm_eps,
                                                 attention.norm_weight_offset);
                }
                if (rr.scalar_type() == mfq_tensor_backend::kBFloat16 &&
                    oo2.scalar_type() == mfq_tensor_backend::kBFloat16) {
                    if (attention.official_bf16) {
                        return acc_rms_norm_bf16_cuda(rr, oo2, ffn_norm, attention.rms_norm_eps,
                                                      attention.norm_weight_offset);
                    }
                    auto sum = (rr + oo2).contiguous();
                    auto norm = qwen_rms_norm(sum.to(mfq_tensor_backend::kFloat32), ffn_norm,
                                              attention.rms_norm_eps, attention.norm_weight_offset)
                                    .to(mfq_tensor_backend::kBFloat16)
                                    .contiguous();
                    return std::vector<mfq_tensor_backend::Tensor>{sum, norm};
                }
                return acc_rms_norm_cuda(rr, oo2, ffn_norm, attention.rms_norm_eps, attention.norm_weight_offset);
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
                    if (!attention.official_bf16) {
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
                    if (attention.official_bf16) {
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

mfq_tensor_backend::Tensor linear_ffn(
    CudaExecutionContext& execution, const FFN& ffn,
    const mfq_tensor_backend::Tensor& ffn_norm, const LinearAttention& attention,
    mfq_tensor_backend::Tensor residual, mfq_tensor_backend::Tensor attention_output) {
    const int64_t B = residual.size(0), T = residual.size(1), H = residual.size(2);
    if (!residual.is_cuda()) {
        return mfq::models::residual_ffn(
            std::move(residual), std::move(attention_output),
            [&](mfq_tensor_backend::Tensor residual,
                mfq_tensor_backend::Tensor attention_output) {
              auto x = (residual.to(mfq_tensor_backend::kFloat32) +
                        attention_output.to(mfq_tensor_backend::kFloat32))
                           .to(residual.scalar_type())
                           .contiguous();
              residual = x;
              auto xn =
                  qwen_rms_norm(
                      x.reshape({B * T, H}).to(mfq_tensor_backend::kFloat32),
                      ffn_norm, attention.rms_norm_eps,
                      attention.norm_weight_offset)
                      .reshape({B, T, H});
              return std::array<mfq_tensor_backend::Tensor, 2>{residual, xn};
            },
            [&](mfq_tensor_backend::Tensor residual,
                mfq_tensor_backend::Tensor xn) {
              return mfq::models::feed_forward(
                  std::move(residual), std::move(xn),
                  [](const auto &, const auto &) {
                    return std::optional<mfq_tensor_backend::Tensor>{};
                  },
                  [&](mfq_tensor_backend::Tensor xn) {
                    return ffn.forward(execution, xn.reshape({B * T, H}))
                        .reshape({B, T, H});
                  },
                  [&](mfq_tensor_backend::Tensor residual,
                      mfq_tensor_backend::Tensor ff) {
                    return (residual.to(mfq_tensor_backend::kFloat32) +
                            ff.to(mfq_tensor_backend::kFloat32))
                        .to(residual.scalar_type())
                        .contiguous();
                  });
            });

    }
    auto& profiler = execution.profiler;
    auto oo = std::move(attention_output);

    auto attn_pair =
        profiler.measure("linear.attn_residual_ffn_norm", [&]() {
          auto rr = residual.reshape({-1, H});
          auto oo2 = oo.reshape({-1, H});
          if (oo2.scalar_type() != rr.scalar_type()) {
            oo2 = oo2.to(rr.scalar_type()).contiguous();
          }
          if (execution.config.diagnostic_fp32_residual) {
            return acc_rms_norm_cuda(
                rr.to(mfq_tensor_backend::kFloat32),
                oo2.to(mfq_tensor_backend::kFloat32), ffn_norm,
                attention.rms_norm_eps,
                attention.norm_weight_offset);
          }
          if (rr.scalar_type() == mfq_tensor_backend::kFloat16 &&
              oo2.scalar_type() == mfq_tensor_backend::kFloat16) {
            return acc_rms_norm_f16_cuda(
                rr, oo2, ffn_norm, attention.rms_norm_eps,
                attention.norm_weight_offset);
          }
          return acc_rms_norm_cuda(
              rr, oo2, ffn_norm, attention.rms_norm_eps,
              attention.norm_weight_offset);
        });

    residual = attn_pair[0].reshape({B, T, H});
    auto xn = attn_pair[1].reshape({B, T, H});
    return mfq::models::feed_forward(
        std::move(residual), std::move(xn),
        [&](mfq_tensor_backend::Tensor xn, mfq_tensor_backend::Tensor residual)
            -> std::optional<mfq_tensor_backend::Tensor> {
          auto ffn_input = xn.reshape({B * T, H});
          auto residual_flat = residual.reshape({B * T, H});
          if (ffn.can_forward_fused_residual(execution.config, ffn_input,
                                             residual_flat)) {
            return profiler.measure("linear.ffn_down_residual", [&]() {
              return ffn
                  .forward_fused_residual(profiler, execution.config, ffn_input,
                                          residual_flat)
                  .reshape({B, T, H});
            });
          }
          return {};
        },
        [&](mfq_tensor_backend::Tensor xn) {
          return ffn.forward(execution, xn.reshape({B * T, H}))
              .reshape({B, T, H});
        },
        [&](mfq_tensor_backend::Tensor residual,
            mfq_tensor_backend::Tensor ff) {
          return profiler.measure("linear.ffn_residual", [&]() {
            auto rr = residual.reshape({-1, H});
            auto ff2 = ff.reshape({-1, H});
            if (execution.config.diagnostic_fp32_residual) {
              rr = rr.to(mfq_tensor_backend::kFloat32);
              ff2 = ff2.to(mfq_tensor_backend::kFloat32);
            } else if (ff2.scalar_type() != rr.scalar_type()) {
              ff2 = ff2.to(rr.scalar_type()).contiguous();
            }
            return acc_cuda(rr, ff2).reshape({B, T, H});
          });
        });

}

} // namespace mfq::cuda
