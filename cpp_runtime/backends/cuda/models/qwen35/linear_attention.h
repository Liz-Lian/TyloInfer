#pragma once
#include "mfq_cuda_linear_attention_ops.h"
#include "mfq_cuda_gdn_replay_ops.h"
#include "mfq_cuda_norm_ops.h"
#include "models/common/transformer_layer.h"
#include "core/recurrent_mixer_state.h"

#include "core/block.h"
#include "core/ffn.h"
#include "gguf_prefill_fusion.h"
#include "models/qwen35/config.h"
#include "models/qwen35/causal_lm.h"
#include <cmath>

namespace mfq::cuda::qwen35 {

struct LinearRecurrentInputs {
  mfq_tensor_backend::Tensor qk;
  mfq_tensor_backend::Tensor value;
  mfq_tensor_backend::Tensor qkv;
  mfq_tensor_backend::Tensor gate;
  mfq_tensor_backend::Tensor beta;
  // Retain the exact recurrent inputs already consumed by verification.
  // Rollback needs a shorter prefix, never another convolution or Q/K norm.
  mfq_tensor_backend::Tensor normalized_q, normalized_k, convolved_v;
  int64_t batch = 0;
  int64_t tokens = 0;
  bool split = false;
  // Fused windows retain rank-one update records in value and raw QKV for
  // convolution history. Prefix replay needs no repeated convolution/norm.
  bool fused = false;
};

struct LinearAttentionBlock final : ::Block {
  mfq::models::qwen35::Config qwen_config;
  mfq_tensor_backend::Tensor attn_norm, ffn_norm, conv_weight, conv_bias,
      dt_bias, a_log, linear_norm;
  bool split_in_proj = false;
  bool split_dense_zab = false;
  bool dense_ab_tail = false;
  bool ab_is_nint = false;
  bool dense_out_proj = false;
  bool tiled_v_heads = false;
  QuantLinearGroup in_proj;
  QuantLinearGroup qkv_proj;
  QuantLinearGroup qkvz_proj;
  QuantLinear z_proj;
  QuantLinearGroup ab_nint_proj;
  DenseLinearGroup ab_proj;
  std::shared_ptr<CudaIndependentBranchExecutor> dense_projection_executor =
      std::make_shared<CudaIndependentBranchExecutor>();
  DenseLinearGroup zab_proj;
  QuantLinear out_proj;
  mfq_tensor_backend::Tensor out_proj_dense;
  FFN ffn;
  mfq::cuda::RecurrentMixerState mixer;
  bool transposed_gdn_state = true;
  int64_t speculative_ffn_batches = 0;
  int64_t speculative_projection_batches = 0;
  LinearRecurrentInputs speculative_inputs;
  int64_t speculative_start = -1;
  int64_t speculative_confirmed = 0;
  int64_t speculative_tokens = 0;

  bool supports_speculation() const noexcept override { return true; }
  std::vector<mfq_tensor_backend::Tensor*> graph_warmup_state() override {
    TYLOINFER_RUNTIME_CHECK(!mixer.speculative_pending && mixer.convolution.defined() &&
                          mixer.recurrent.defined(),
                      "decode warmup requires confirmed recurrent state");
    return {&mixer.convolution, &mixer.recurrent};
  }
  mfq_tensor_backend::Tensor forward_context(CudaExecutionContext &execution,
                                             mfq_tensor_backend::Tensor input,
                                             const Block::Context &context,
                                             const RopeCache &rope) override;
  void commit_speculative() override;
  void rollback_speculative(int64_t keep_position) override;

  void reset(int64_t B) override {
    clear_speculative();
    if(decode_core_workspace.defined())decode_core_workspace.zero_();
    if (mixer.convolution.defined() && mixer.recurrent.defined() &&
        mixer.convolution.size(0) == B && mixer.recurrent.size(0) == B) {
      mixer.convolution.zero_();
      mixer.recurrent.zero_();
      return;
    }
    mixer.convolution = mfq_tensor_backend::Tensor();
    mixer.recurrent = mfq_tensor_backend::Tensor();
  }

  void clear_speculative() noexcept;

  mfq_tensor_backend::Tensor forward_cpu(CudaExecutionContext &execution,
                                         mfq_tensor_backend::Tensor x) {
    const int64_t B = x.size(0), T = x.size(1), H = x.size(2);
    const int64_t nk = qwen_config.linear_num_key_heads;
    const int64_t nv = qwen_config.linear_num_value_heads;
    const int64_t dk = qwen_config.linear_key_head_dim;
    const int64_t dv = qwen_config.linear_value_head_dim;
    const int64_t ksz = qwen_config.linear_k_size();
    const int64_t vsz = qwen_config.linear_v_size();
    const int64_t conv_dim = 2 * ksz + vsz;
    if (!mixer.convolution.defined()) {
      auto f32 = mfq_tensor_backend::TensorOptions()
                     .device(mfq_tensor_backend::kCPU)
                     .dtype(mfq_tensor_backend::kFloat32);
      mixer.convolution = mfq_tensor_backend::zeros(
          {B, qwen_config.linear_conv_kernel_dim - 1, conv_dim}, f32);
      mixer.recurrent = mfq_tensor_backend::zeros({B, nv, dv, dv}, f32);
    }

    return mfq::models::attention_layer(
        std::move(x),
        [&](mfq_tensor_backend::Tensor x) {
          return qwen_rms_norm(
                     x.reshape({B * T, H}),
                     attn_norm, qwen_config.rms_norm_eps,
                     qwen_config.legacy_tensor_layout.norm_weight_offset)
              .reshape({B, T, H});
        },
        [&](mfq_tensor_backend::Tensor xn) {
          return mfq::models::qwen35::linear_attention(
              std::move(xn),
              [&](mfq_tensor_backend::Tensor xn) {
                mfq_tensor_backend::Tensor qkv, qk_part, v_part, z, alpha_raw,
                    beta_raw;
                if (dense_ab_tail) {
                  auto parts = in_proj.forward(execution, xn);
                  qkv = parts[0].to(mfq_tensor_backend::kFloat32);
                  z = parts[1];
                  auto ab = ab_proj.forward(xn);
                  alpha_raw = ab[0];
                  beta_raw = ab[1];
                } else if (split_in_proj) {
                  if (split_dense_zab) {
                    auto qkv_parts = qkv_proj.forward(execution, xn);
                    qk_part = qkv_parts[0];
                    v_part = qkv_parts[1];
                    auto zab = zab_proj.forward(xn);
                    z = zab[0];
                    alpha_raw = zab[1];
                    beta_raw = zab[2];
                  } else {
                    auto qkv_parts = qkv_proj.forward(execution, xn);
                    qk_part = qkv_parts[0];
                    v_part = qkv_parts[1];
                    z = z_proj.forward(execution, xn);
                    auto ab = ab_is_nint ? ab_nint_proj.forward(execution, xn)
                                         : ab_proj.forward(xn);
                    alpha_raw = ab[0];
                    beta_raw = ab[1];
                  }
                  qkv = mfq_tensor_backend::cat(
                      {qk_part.to(mfq_tensor_backend::kFloat32),
                       v_part.to(mfq_tensor_backend::kFloat32)},
                      -1);
                } else {
                  auto parts = in_proj.forward(execution, xn);
                  qkv = parts[0].to(mfq_tensor_backend::kFloat32);
                  z = parts[1];
                  alpha_raw = parts[2];
                  beta_raw = parts[3];
                }

                return mfq::models::qwen35::LinearProjections<
                    mfq_tensor_backend::Tensor>{qkv, qk_part,   v_part,
                                                z,   alpha_raw, beta_raw};
              },
              [&](const mfq::models::qwen35::LinearProjections<
                  mfq_tensor_backend::Tensor> &p) {
                auto alpha_raw = p.alpha, beta_raw = p.beta;
                auto beta = mfq_tensor_backend::sigmoid(
                    beta_raw.to(mfq_tensor_backend::kFloat32)
                        .reshape({B, T, nv}));
                auto alpha = alpha_raw.to(mfq_tensor_backend::kFloat32)
                                 .reshape({B, T, nv});
                auto gate = mfq_tensor_backend::softplus(
                    alpha + dt_bias.to(mfq_tensor_backend::kFloat32)
                                .reshape({1, 1, nv}));
                auto coefficient = -mfq_tensor_backend::exp(
                    a_log.to(mfq_tensor_backend::kFloat32));
                gate = gate * coefficient.reshape({1, 1, nv});

                auto gate_t = gate.transpose(1, 2).contiguous();
                auto beta_t = beta.transpose(1, 2).contiguous();

                return std::array<mfq_tensor_backend::Tensor, 2>{gate_t,
                                                                 beta_t};
              },
              [&](const mfq::models::qwen35::LinearProjections<
                  mfq_tensor_backend::Tensor> &p) {
                auto qkv = p.qkv, qk_part = p.qk, v_part = p.value;
                auto conv_input = mfq_tensor_backend::cat({mixer.convolution, qkv}, 1);
                mixer.convolution.copy_(conv_input.narrow(
                    1,
                    conv_input.size(1) -
                        (qwen_config.linear_conv_kernel_dim - 1),
                    qwen_config.linear_conv_kernel_dim - 1));
                auto weight = conv_weight.to(mfq_tensor_backend::kFloat32);
                if (weight.dim() == 3)
                  weight = weight.squeeze(1);
                if (weight.dim() != 2) {
                  throw std::runtime_error(
                      "CPU SSM convolution weight must be rank 2 or 3");
                }
                if (weight.size(0) != conv_dim && weight.size(1) == conv_dim) {
                  weight = weight.transpose(0, 1).contiguous();
                }
                TYLOINFER_RUNTIME_CHECK(weight.size(0) == conv_dim &&
                                      weight.size(1) ==
                                          qwen_config.linear_conv_kernel_dim,
                                  "CPU SSM convolution geometry mismatch");
                auto windows =
                    conv_input.unfold(1, qwen_config.linear_conv_kernel_dim, 1);
                auto conv =
                    (windows *
                     weight.reshape(
                         {1, 1, conv_dim, qwen_config.linear_conv_kernel_dim}))
                        .sum(-1);
                if (conv_bias.defined()) {
                  conv = conv + conv_bias.to(mfq_tensor_backend::kFloat32)
                                    .reshape({1, 1, conv_dim});
                }
                conv = mfq_tensor_backend::silu(conv);

                auto q = conv.narrow(-1, 0, ksz)
                             .reshape({B, T, nk, dk})
                             .transpose(1, 2)
                             .contiguous();
                auto k = conv.narrow(-1, ksz, ksz)
                             .reshape({B, T, nk, dk})
                             .transpose(1, 2)
                             .contiguous();
                auto v = conv.narrow(-1, 2 * ksz, vsz)
                             .reshape({B, T, nv, dv})
                             .transpose(1, 2)
                             .contiguous();
                q = q / mfq_tensor_backend::sqrt(q.square().sum(-1, true))
                            .clamp_min(qwen_config.rms_norm_eps);
                k = k / mfq_tensor_backend::sqrt(k.square().sum(-1, true))
                            .clamp_min(qwen_config.rms_norm_eps);
                if (nk != nv) {
                  TYLOINFER_RUNTIME_CHECK(nv % nk == 0,
                                    "CPU GDN head ratio is invalid");
                  const int64_t repeat = nv / nk;
                  if (tiled_v_heads) {
                    q = q.repeat({1, repeat, 1, 1});
                    k = k.repeat({1, repeat, 1, 1});
                  } else {
                    q = q.repeat_interleave(repeat, 1);
                    k = k.repeat_interleave(repeat, 1);
                  }
                }

                return std::array<mfq_tensor_backend::Tensor, 3>{q, k, v};
              },
              [&](std::array<mfq_tensor_backend::Tensor, 3> qkv,
                  std::array<mfq_tensor_backend::Tensor, 2> gates) {
                auto q = qkv[0], k = qkv[1], v = qkv[2];
                auto gate_t = gates[0], beta_t = gates[1];
                auto state = mixer.recurrent;
                std::vector<mfq_tensor_backend::Tensor> outputs;
                outputs.reserve(static_cast<size_t>(T));
                const double retrieve_scale =
                    1.0 / std::sqrt(static_cast<double>(dk));
                for (int64_t token = 0; token < T; ++token) {
                  auto qt = q.select(2, token);
                  auto kt = k.select(2, token);
                  auto vt = v.select(2, token);
                  state =
                      state * mfq_tensor_backend::exp(gate_t.select(2, token))
                                  .reshape({B, nv, 1, 1});
                  auto state_k = mfq_tensor_backend::matmul(
                                     state.transpose(-1, -2), kt.unsqueeze(-1))
                                     .squeeze(-1);
                  auto delta =
                      (vt - state_k) * beta_t.select(2, token).unsqueeze(-1);
                  state = state + kt.unsqueeze(-1) * delta.unsqueeze(-2);
                  outputs.push_back(
                      mfq_tensor_backend::matmul(state.transpose(-1, -2),
                                                 qt.unsqueeze(-1))
                          .squeeze(-1) *
                      retrieve_scale);
                }
                mixer.recurrent = state.contiguous();
                auto y = mfq_tensor_backend::stack(outputs, 2);

                return y;
              },
              [&](mfq_tensor_backend::Tensor y) {
                auto y_flat =
                    y.reshape({-1, dv}).to(mfq_tensor_backend::kFloat32);
                auto inverse = mfq_tensor_backend::rsqrt(
                    y_flat.square().mean(-1, true) + qwen_config.rms_norm_eps);
                auto y_norm = (y_flat * inverse *
                               linear_norm.to(mfq_tensor_backend::kFloat32))
                                  .reshape_as(y);
                auto yf =
                    y_norm.transpose(1, 2).contiguous().reshape({B, T, vsz});

                return yf;
              },
              [&](mfq_tensor_backend::Tensor yf, mfq_tensor_backend::Tensor z) {
                z = z.reshape({B, T, nv, dv}).transpose(1, 2).contiguous();
                auto zf = z.transpose(1, 2).contiguous().reshape({B, T, vsz});
                mfq_tensor_backend::Tensor attention_output;
                if (dense_out_proj) {
                  auto dtype = out_proj_dense.scalar_type();
                  auto gated =
                      yf.to(dtype) * mfq_tensor_backend::silu(zf.to(dtype));
                  attention_output =
                      dense_projection(gated.reshape({B * T, vsz}),
                                       out_proj_dense)
                          .reshape({B, T, H});
                } else {
                  attention_output =
                      out_proj.forward_input_mul(execution, yf, zf, 2);
                }
                return attention_output;
              });
        },
        [&](mfq_tensor_backend::Tensor residual,
            mfq_tensor_backend::Tensor attention_output) {
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
                        x.reshape({B * T, H}),
                        ffn_norm, qwen_config.rms_norm_eps,
                        qwen_config.legacy_tensor_layout.norm_weight_offset)
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
        });
  }

  std::array<mfq_tensor_backend::Tensor, 2>
  forward_attention_cuda(CudaExecutionContext &execution,
                         mfq_tensor_backend::Tensor x,
                         LinearRecurrentInputs *captured = nullptr,
                         GgufProjectionInput* normalized_ffn_input=nullptr) {
    auto &profiler = execution.profiler;
    transposed_gdn_state = execution.config.gdn_transposed_state;
    int64_t B = x.size(0), T = x.size(1), H = x.size(2);
    int64_t nk = qwen_config.linear_num_key_heads,
            nv = qwen_config.linear_num_value_heads;
    int64_t dk = qwen_config.linear_key_head_dim,
            dv = qwen_config.linear_value_head_dim;
    int64_t ksz = qwen_config.linear_k_size(),
            vsz = qwen_config.linear_v_size();
    int64_t conv_dim = 2 * ksz + vsz;
    if (!mixer.convolution.defined()) {
      mixer.convolution = mfq_tensor_backend::zeros(
          {B, qwen_config.linear_conv_kernel_dim - 1, conv_dim},
          mfq_tensor_backend::TensorOptions()
              .device(mfq_tensor_backend::kCUDA)
              .dtype(mfq_tensor_backend::kFloat32));
      mixer.recurrent = mfq_tensor_backend::zeros(
          {B, nv, dv, dv}, mfq_tensor_backend::TensorOptions()
                               .device(mfq_tensor_backend::kCUDA)
                               .dtype(mfq_tensor_backend::kFloat32));
    }
    GgufProjectionInput normalized_projection_input;
    std::unique_ptr<GgufPrefillInputScope> normalized_prefill_input;
    GgufPrefillProduct scaled_output;
    const bool scaled_residual=T>=512 && x.scalar_type()==mfq_tensor_backend::kFloat16 &&
        execution.kl_mmq.mode==KlMmqMode::Default && !execution.config.diagnostic_fp32_residual &&
        ffn_norm.scalar_type()==mfq_tensor_backend::kFloat32 && ffn_norm.is_contiguous() && ffn_norm.numel()==H;

    return mfq::models::attention_layer(
        std::move(x),
        [&](mfq_tensor_backend::Tensor x) {
          return profiler.measure("linear.attn_norm", [&]() {
            if(dense_ab_tail || !split_in_proj) {
              mfq_tensor_backend::Tensor normalized;
              normalized_prefill_input=in_proj.prepare_prefill_rms_norm_input(execution,x,attn_norm,
                  qwen_config.rms_norm_eps,qwen_config.legacy_tensor_layout.norm_weight_offset,normalized);
              if(normalized.defined())return normalized;
              normalized_projection_input=in_proj.prepare_rms_norm_input(execution,x,attn_norm,
                  qwen_config.rms_norm_eps,qwen_config.legacy_tensor_layout.norm_weight_offset,normalized);
              if(normalized.defined())return normalized;
            }
            return qwen_rms_norm(
                       x.reshape({B * T, H}),
                       attn_norm, qwen_config.rms_norm_eps,
                       qwen_config.legacy_tensor_layout.norm_weight_offset)
                .reshape({B, T, H});
          });
        },
        [&](mfq_tensor_backend::Tensor xn) {
          const auto taps=qwen_config.linear_conv_kernel_dim;
          const bool core_conv_layout=
              (conv_weight.dim()==2 &&
               ((conv_weight.size(0)==conv_dim && conv_weight.size(1)==taps) ||
                (conv_weight.size(0)==taps && conv_weight.size(1)==conv_dim))) ||
              (conv_weight.dim()==3 && conv_weight.size(0)==conv_dim &&
               conv_weight.size(1)==1 && conv_weight.size(2)==taps);
          const bool packed_convolution_candidate=T>1 && !split_in_proj &&
              execution.config.linear_conv_prefill_fused && dk==dv && dk>0 && dk<=256 && nk>0 &&
              nv%nk==0 && taps>=2 && taps<=8 && core_conv_layout &&
              (!conv_bias.defined() || conv_bias.numel()==0) &&
              conv_weight.scalar_type()==mfq_tensor_backend::kFloat32 && conv_weight.is_contiguous();
          GgufPrefillProduct scaled_qkv,scaled_gate;
          const auto project_inputs=[&](mfq_tensor_backend::Tensor xn) {
                const auto project_group=[&] {
                  if(T>=256 && packed_convolution_candidate && execution.kl_mmq.mode==KlMmqMode::Default &&
                      normalized_projection_input.shape.empty() && !in_proj.layers.empty() &&
                      std::all_of(in_proj.layers.begin(),in_proj.layers.end(),[](const QuantLinear& linear) {
                        return linear.is_gguf() && !linear.tensor_parallel() && !linear.gguf.prefers_original_input();
                      })) {
                    GgufPrefillInputScope input_scope(xn);
                    scaled_qkv=gguf_prefill_product(in_proj.layers[0].gguf,xn);
                    if(scaled_qkv.defined()) {
                      std::vector<mfq_tensor_backend::Tensor> parts;
                      parts.reserve(in_proj.layers.size());
                      parts.push_back(scaled_qkv.values.reshape({B,T,in_proj.layers[0].out()}));
                      for(size_t i=1;i<in_proj.layers.size();++i) {
                        if(i==1 && scaled_residual && !dense_out_proj && out_proj.is_gguf() && !out_proj.tensor_parallel())
                          scaled_gate=gguf_prefill_product(in_proj.layers[i].gguf,xn);
                        parts.push_back(i==1 && scaled_gate.defined()
                            ?scaled_gate.values.reshape({B,T,in_proj.layers[i].out()})
                            :in_proj.layers[i].forward(execution,xn));
                      }
                      return parts;
                    }
                  }
                  return normalized_projection_input.shape.empty()?in_proj.forward(execution,xn)
                      :in_proj.forward_prepared_rms(normalized_projection_input);
                };
                mfq_tensor_backend::Tensor qkv, qk_part, v_part, z, alpha_raw,
                    beta_raw;
                if (dense_ab_tail) {
                  std::vector<mfq_tensor_backend::Tensor> parts,ab;
                  const auto project=[&] {
                    parts=profiler.measure("linear.in_proj", [&]() {
                      return project_group();
                    });
                  };
                  if(!profiler.enabled)ab=ab_proj.forward_parallel(xn,project,*dense_projection_executor);
                  else {
                    project();ab=profiler.measure("linear.ab_proj",[&]{return ab_proj.forward(xn);});
                  }
                  qkv = profiler.measure("linear.qkv_cast", [&]() {
                    return scaled_qkv.defined()?parts[0]:parts[0].to(mfq_tensor_backend::kFloat32);
                  });
                  z = parts[1];
                  alpha_raw = ab[0];
                  beta_raw = ab[1];
                } else if (split_in_proj) {
                  bool shared_projection_input = false;
                  if (!shared_projection_input && !split_dense_zab &&
                      ab_is_nint &&
                      execution.config.tensor_parallel_grouped_projections &&
                      execution.config
                          .tensor_parallel_shared_linear_attention_input &&
                      qkv_proj.layers.size() == 2) {
                    QuantLinearProjectionRefs projections = {
                        &qkv_proj.layers[0],
                        &qkv_proj.layers[1],
                        &z_proj,
                    };
                    const bool shared_tp_ab =
                        ab_nint_proj.layers.size() == 2 &&
                        ab_nint_proj.layers[0].tensor_parallel() &&
                        ab_nint_proj.layers[1].tensor_parallel();
                    if (shared_tp_ab) {
                      projections.push_back(&ab_nint_proj.layers[0]);
                      projections.push_back(&ab_nint_proj.layers[1]);
                    }
                    if (tensor_parallel_output_projections_compatible(
                            projections)) {
                      auto parts =
                          profiler.measure("linear.split_projections", [&]() {
                            return forward_tensor_parallel_output_projections(
                                execution, xn, projections);
                          });
                      qk_part = parts[0];
                      v_part = parts[1];
                      z = parts[2];
                      if (shared_tp_ab) {
                        alpha_raw = parts[3];
                        beta_raw = parts[4];
                      } else {
                        auto ab = profiler.measure("linear.ab_proj", [&]() {
                          return ab_nint_proj.forward(execution, xn);
                        });
                        alpha_raw = ab[0];
                        beta_raw = ab[1];
                      }
                      shared_projection_input = true;
                    }
                  }
                  if (!shared_projection_input) {
                    if (split_dense_zab) {
                      auto qkv_parts =
                          profiler.measure("linear.qkv_proj", [&]() {
                            return qkv_proj.forward(execution, xn);
                          });
                      qk_part = qkv_parts[0];
                      v_part = qkv_parts[1];
                      auto zab = profiler.measure("linear.zab_proj", [&]() {
                        return zab_proj.forward(xn);
                      });
                      z = zab[0];
                      alpha_raw = zab[1];
                      beta_raw = zab[2];
                    } else {
                      auto qkv_parts =
                          profiler.measure("linear.qkv_proj", [&]() {
                            return qkv_proj.forward(execution, xn);
                          });
                      qk_part = qkv_parts[0];
                      v_part = qkv_parts[1];
                      z = profiler.measure("linear.z_proj", [&]() {
                        return z_proj.forward(execution, xn);
                      });
                      auto ab = profiler.measure("linear.ab_proj", [&]() {
                        return ab_is_nint ? ab_nint_proj.forward(execution, xn)
                                          : ab_proj.forward(xn);
                      });
                      alpha_raw = ab[0];
                      beta_raw = ab[1];
                    }
                  }
                } else {
                  auto parts = profiler.measure("linear.in_proj", [&]() {
                    return project_group();
                  });
                  qkv = profiler.measure("linear.qkv_cast", [&]() {
                    return scaled_qkv.defined()?parts[0]:parts[0].to(mfq_tensor_backend::kFloat32);
                  });
                  z = parts[1];
                  alpha_raw = parts[2];
                  beta_raw = parts[3];
                }

                return mfq::models::qwen35::LinearProjections<
                    mfq_tensor_backend::Tensor>{qkv, qk_part,   v_part,
                                                z,   alpha_raw, beta_raw};
          };
          const auto projected=project_inputs(xn);
          normalized_prefill_input.reset();
          // Decode and short verification share the fused mixer arithmetic.
          // Verification keeps confirmed recurrence unchanged until commit.
          const bool fused_core_inputs=execution.kl_mmq.mode==KlMmqMode::Default && !split_in_proj &&
              taps>1 && core_conv_layout && std::isfinite(qwen_config.rms_norm_eps) &&
              qwen_config.rms_norm_eps>0 && dk==dv && (dk==32 || dk==64 || dk==128) &&
              (!conv_bias.defined() || conv_bias.numel()==0) && !dense_out_proj &&
              out_proj.is_gguf() && !out_proj.tensor_parallel() &&
              projected.qkv.defined() && projected.output_gate.scalar_type()==projected.alpha.scalar_type() &&
              projected.alpha.scalar_type()==projected.beta.scalar_type() &&
              (projected.alpha.scalar_type()==mfq_tensor_backend::kFloat16 ||
               projected.alpha.scalar_type()==mfq_tensor_backend::kFloat32) &&
              conv_weight.scalar_type()==mfq_tensor_backend::kFloat32 && conv_weight.is_contiguous() &&
              dt_bias.scalar_type()==mfq_tensor_backend::kFloat32 && dt_bias.is_contiguous() &&
              a_log.scalar_type()==mfq_tensor_backend::kFloat32 && a_log.is_contiguous() &&
              linear_norm.scalar_type()==mfq_tensor_backend::kFloat32 && linear_norm.is_contiguous();
          if(fused_core_inputs && captured && B==1 && T>=2 && T<=8 && taps==4 && dk==128) {
            const auto& p=projected;
            auto weight=conv_weight;
            if(weight.dim()==2 && weight.size(0)==conv_dim && weight.size(1)==taps)
              weight=weight.reshape({conv_dim,1,taps});
            const bool quantized=!out_proj.gguf.prefers_original_input();
            const bool position_major=quantized && out_proj.gguf.prefers_position_major_q8(T);
            auto result=profiler.measure("linear.verify_core",[&] {
              return gdn_verify_window_cuda(p.qkv,p.output_gate,p.alpha.reshape({B,T,nv}),p.beta.reshape({B,T,nv}),
                mixer.convolution,mixer.recurrent,weight,dt_bias,a_log,linear_norm,nk,nv,
                qwen_config.rms_norm_eps,qwen_config.rms_norm_eps,true,false,transposed_gdn_state,
                tiled_v_heads,quantized,position_major);
            });
            captured->qkv=p.qkv;captured->value=result[3];
            captured->batch=B;captured->tokens=T;captured->split=false;captured->fused=true;
            mixer.convolution.copy_(result[1]);mixer.speculative_recurrent=result[2];
            return profiler.measure("linear.out_proj_prepared",[&] {
              if(!quantized)return out_proj.forward(execution,result[0].reshape({B,T,vsz}));
              return out_proj.gguf.forward_q8(result[0],T,mfq_tensor_backend::kFloat32,position_major).reshape({B,T,H});
            });
          }
          if(fused_core_inputs && T==1 && !captured) {
            const auto& p=projected;
            if(!decode_core_workspace.defined() || decode_core_workspace.sizes().vec()!=std::vector<int64_t>({B,nv,dv+1}))
              decode_core_workspace=mfq_tensor_backend::zeros({B,nv,dv+1},mixer.recurrent.options());
            auto weight=conv_weight;
            if(weight.dim()==2 && weight.size(0)==conv_dim && weight.size(1)==qwen_config.linear_conv_kernel_dim)
              weight=weight.reshape({conv_dim,1,qwen_config.linear_conv_kernel_dim});
            // Activation-dependent corrections retain the floating readout;
            // ordinary integer projections can consume Q8 directly.
            const bool quantized_readout=!out_proj.gguf.prefers_original_input();
            auto result=profiler.measure(quantized_readout?"linear.decode_core_q8":"linear.decode_core_f32",[&] {
              return gdn_decode_core_cuda(p.qkv,p.output_gate,p.alpha.reshape({B,1,nv}),p.beta.reshape({B,1,nv}),
                  mixer.convolution,mixer.recurrent,weight,dt_bias,a_log,linear_norm,nk,nv,dv,
                  qwen_config.rms_norm_eps,qwen_config.rms_norm_eps,true,false,transposed_gdn_state,
                  decode_core_workspace,true,{},tiled_v_heads,quantized_readout);
            });
            // The core can update convolution in place or return fresh history.
            if(result[1].data_ptr()!=mixer.convolution.data_ptr())mixer.convolution.copy_(result[1]);
            return profiler.measure("linear.out_proj_prepared",[&] {
              if(!quantized_readout)return out_proj.forward(execution,result[0].reshape({B,1,vsz}));
              return out_proj.gguf.forward_q8(result[0],B,mfq_tensor_backend::kFloat32).reshape({B,1,H});
            });
          }
          std::vector<mfq_tensor_backend::Tensor> packed_prefill;
          const auto row_strided=[&](const mfq_tensor_backend::Tensor& value,int64_t columns) {
            return value.defined() && value.is_cuda() && value.dim()==3 &&
                value.size(0)==B && value.size(1)==T && value.size(2)==columns &&
                value.stride(2)==1 && value.stride(1)>=columns &&
                (B==1 || value.stride(0)>=T*value.stride(1)) &&
                (value.scalar_type()==mfq_tensor_backend::kFloat16 ||
                 value.scalar_type()==mfq_tensor_backend::kFloat32);
          };
          if(packed_convolution_candidate &&
              row_strided(projected.qkv,conv_dim) && row_strided(projected.alpha,nv) &&
              row_strided(projected.beta,nv) && projected.alpha.scalar_type()==projected.beta.scalar_type() &&
              conv_weight.scalar_type()==mfq_tensor_backend::kFloat32 && conv_weight.is_contiguous()) {
            auto weight=conv_weight;
            if(weight.dim()==2 && weight.size(0)==conv_dim && weight.size(1)==taps)
              weight=weight.reshape({conv_dim,1,taps});
            packed_prefill=profiler.measure("linear.conv_norm_gates_prefill",[&] {
              if(scaled_qkv.defined())return linear_conv_qkv_gate_scaled_prefill_cuda(mixer.convolution,projected.qkv,
                  scaled_qkv.row_exponents,scaled_qkv.column_exponents,weight,projected.alpha,projected.beta,
                  dt_bias,a_log,nk,nv,dk,qwen_config.rms_norm_eps,false);
              return linear_conv_qkv_gate_prefill_cuda(mixer.convolution,projected.qkv,weight,
                  projected.alpha,projected.beta,dt_bias,a_log,nk,nv,dk,qwen_config.rms_norm_eps,false);
            });
            if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
              static std::atomic<bool> reported{};
              if(!reported.exchange(true))std::fprintf(stderr,
                  "native_gdn_packed_prefill actual_match=1 tokens=%lld fused_conv=1 fused_l2=1 fused_gates=1 direct_head_layout=1\n",
                  static_cast<long long>(T));
            }
          }
          return mfq::models::qwen35::linear_attention(std::move(xn),
              [&](mfq_tensor_backend::Tensor) {return projected;},
              [&](const mfq::models::qwen35::LinearProjections<
                  mfq_tensor_backend::Tensor> &p) {
                const auto remember_gates=[&](mfq_tensor_backend::Tensor gate_t,
                                               mfq_tensor_backend::Tensor beta_t) {
                 if (captured != nullptr) {
                  captured->qk = p.qk;
                  captured->value = p.value;
                  captured->qkv = p.qkv;
                  captured->gate = gate_t;
                  captured->beta = beta_t;
                  captured->batch = B;
                  captured->tokens = T;
                  captured->split = split_in_proj;
                 }
                return std::array<mfq_tensor_backend::Tensor, 2>{gate_t,
                                                                 beta_t};
                };
                if(!packed_prefill.empty())return remember_gates(packed_prefill[4],packed_prefill[5]);
                auto gates = profiler.measure("linear.gates_fused", [&]() {
                  return linear_gate_beta_cuda(p.alpha.reshape({B, T, nv}),
                                               p.beta.reshape({B, T, nv}),
                                               dt_bias, a_log);
                });
                return remember_gates(gates[0],gates[1]);
              },
              [&](const mfq::models::qwen35::LinearProjections<
                  mfq_tensor_backend::Tensor> &p) {
                if(!packed_prefill.empty()) {
                  mixer.convolution.copy_(packed_prefill[3]);
                  return std::array<mfq_tensor_backend::Tensor,3>{packed_prefill[0],packed_prefill[1],packed_prefill[2]};
                }
                auto qkv = p.qkv, qk_part = p.qk, v_part = p.value;
                auto bias =
                    conv_bias.defined()
                        ? conv_bias
                        : mfq_tensor_backend::empty(
                              {0}, mfq_tensor_backend::TensorOptions()
                                       .device(mfq_tensor_backend::kCUDA)
                                       .dtype(mfq_tensor_backend::kFloat32));
                mfq_tensor_backend::Tensor q, k, v;
                bool fused_prefill =
                    T >= 256 && execution.config.linear_conv_prefill_fused;
                if (T > 1 && split_in_proj && fused_prefill) {
                  auto qkv_fast =
                      profiler.measure("linear.conv_qkv_prefill", [&]() {
                        return linear_conv_qkv_prefill_cuda(
                            mixer.convolution, qk_part.contiguous(),
                            v_part.contiguous(), conv_weight, bias, nk, nv, dk,
                            dv, qwen_config.rms_norm_eps);
                      });
                  q = qkv_fast[0];
                  k = qkv_fast[1];
                  v = qkv_fast[2];
                  // Runtime CUDA graphs retain this state storage address
                  // across requests.
                  mixer.convolution.copy_(qkv_fast[3]);
                } else if (T == 1 && !split_in_proj && dk <= 256) {
                  auto qkv_fast = profiler.measure("linear.conv_qkv_decode_f32", [&]() {
                    return linear_conv_qkv_decode_f32_cuda(
                        mixer.convolution, qkv.contiguous(), conv_weight, bias,
                        nk, nv, dk, dv, qwen_config.rms_norm_eps);
                  });
                  q = qkv_fast[0];
                  k = qkv_fast[1];
                  v = qkv_fast[2];
                } else if (T == 1 && split_in_proj) {
                  auto qkv_fast =
                      profiler.measure("linear.conv_qkv_decode", [&]() {
                        return linear_conv_qkv_decode_cuda(
                            mixer.convolution, qk_part.contiguous(),
                            v_part.contiguous(), conv_weight, bias, nk, nv, dk,
                            dv, qwen_config.rms_norm_eps);
                      });
                  q = qkv_fast[0];
                  k = qkv_fast[1];
                  v = qkv_fast[2];
                } else {
                  if (split_in_proj) {
                    qkv = profiler.measure("linear.qkv_cat", [&]() {
                      return mfq_tensor_backend::cat(
                          {qk_part.to(mfq_tensor_backend::kFloat32),
                           v_part.to(mfq_tensor_backend::kFloat32)},
                          -1);
                    });
                  }
                  mfq_tensor_backend::Tensor conv;
                  if (T == 1) {
                    conv = profiler.measure("linear.conv_silu_decode", [&]() {
                      return ssm_conv_silu_decode_cuda(mixer.convolution, qkv,
                                                       conv_weight, bias);
                    });
                  } else {
                    auto conv_in = profiler.measure("linear.conv_input", [&]() {
                      return mfq_tensor_backend::cat({mixer.convolution, qkv}, 1);
                    });
                    auto next_conv_state =
                        profiler.measure("linear.conv_state_update", [&]() {
                          return conv_in
                              .index(
                                  {Slice(),
                                   Slice(-(qwen_config.linear_conv_kernel_dim -
                                           1),
                                         mfq_tensor_backend::indexing::None),
                                   Slice()})
                              .contiguous();
                        });
                    mixer.convolution.copy_(next_conv_state);
                    conv = profiler.measure("linear.conv_silu", [&]() {
                      return ssm_conv_silu_cuda(conv_in, conv_weight, bias, T);
                    });
                  }
                  q = profiler.measure("linear.q_view", [&]() {
                    return conv.index({Slice(), Slice(), Slice(0, ksz)})
                        .reshape({B, T, nk, dk})
                        .transpose(1, 2);
                  });
                  k = profiler.measure("linear.k_view", [&]() {
                    return conv.index({Slice(), Slice(), Slice(ksz, 2 * ksz)})
                        .reshape({B, T, nk, dk})
                        .transpose(1, 2);
                  });
                  v = profiler.measure("linear.v_view", [&]() {
                    return conv
                        .index(
                            {Slice(), Slice(), Slice(2 * ksz, 2 * ksz + vsz)})
                        .reshape({B, T, nv, dv})
                        .transpose(1, 2);
                  });
                  q = profiler.measure("linear.q_l2", [&]() {
                    return l2_norm_cuda(q.contiguous().reshape({-1, dk}),
                                        qwen_config.rms_norm_eps)
                        .reshape_as(q);
                  });
                  k = profiler.measure("linear.k_l2", [&]() {
                    return l2_norm_cuda(k.contiguous().reshape({-1, dk}),
                                        qwen_config.rms_norm_eps)
                        .reshape_as(k);
                  });
                }

                return std::array<mfq_tensor_backend::Tensor, 3>{q, k, v};
              },
              [&](std::array<mfq_tensor_backend::Tensor, 3> qkv,
                  std::array<mfq_tensor_backend::Tensor, 2> gates) {
                auto q = qkv[0].contiguous(), k = qkv[1].contiguous(), v = qkv[2].contiguous();
                if(captured) {
                  captured->normalized_q=q;
                  captured->normalized_k=k;
                  captured->convolved_v=v;
                }
                auto gate_t = gates[0], beta_t = gates[1];
                auto recurrent_step = [&](mfq_tensor_backend::Tensor rq,
                                          mfq_tensor_backend::Tensor rk,
                                          mfq_tensor_backend::Tensor rv,
                                          mfq_tensor_backend::Tensor rg,
                                          mfq_tensor_backend::Tensor rb) {
                  const bool transposed = transposed_gdn_state;
                  if(captured) {
                    // Leave committed recurrence at a stable address for the
                    // next graph replay and any rejected-prefix restoration.
                    if(transposed) {
                      if(tiled_v_heads)return gdn_transposed_tiled_cuda(rq,rk,rv,rg,rb,mixer.recurrent);
                      return gdn_transposed_cuda(rq,rk,rv,rg,rb,mixer.recurrent);
                    }
                    if(tiled_v_heads)return gdn_tiled_cuda(rq,rk,rv,rg,rb,mixer.recurrent);
                    return gdn_cuda(rq,rk,rv,rg,rb,mixer.recurrent);
                  }
                  if (transposed) {
                    if (tiled_v_heads)
                      return gdn_inplace_transposed_tiled_cuda(
                          rq.contiguous(), rk.contiguous(), rv.contiguous(), rg,
                          rb, mixer.recurrent);
                    return gdn_inplace_transposed_cuda(
                        rq.contiguous(), rk.contiguous(), rv.contiguous(), rg,
                        rb, mixer.recurrent);
                  }
                  if (tiled_v_heads)
                    return gdn_inplace_tiled_cuda(
                        rq.contiguous(), rk.contiguous(), rv.contiguous(), rg,
                        rb, mixer.recurrent);
                  return gdn_inplace_cuda(rq.contiguous(), rk.contiguous(),
                                          rv.contiguous(), rg, rb, mixer.recurrent);
                };
                auto gd = profiler.measure("linear.gdn", [&]() {
                  return recurrent_step(q, k, v, gate_t, beta_t);
                });
                auto y = gd[0];
                if(captured)mixer.speculative_recurrent=gd[1];
                else mixer.recurrent = gd[1];

                return y;
              },
              [&](mfq_tensor_backend::Tensor y) {
                // QuantLinear can normalize while preparing the projection input.
                if (!dense_out_proj) return y;
                auto y_norm = profiler.measure("linear.out_norm", [&]() {
                  return rms_norm_cuda(y.reshape({-1, dv}).to(
                                           mfq_tensor_backend::kFloat32),
                                       linear_norm, qwen_config.rms_norm_eps)
                      .reshape_as(y);
                });
                auto yf = profiler.measure("linear.y_view", [&]() {
                  return y_norm.transpose(1, 2).contiguous().reshape(
                      {B, T, vsz});
                });

                return yf;
              },
              [&](mfq_tensor_backend::Tensor yf, mfq_tensor_backend::Tensor z) {
                if (!dense_out_proj) {
                  return profiler.measure("linear.out_proj_gate", [&]() {
                    if(scaled_residual && out_proj.is_gguf() && !out_proj.tensor_parallel()) {
                      scaled_output=scaled_gate.defined()
                          ?gguf_rms_gated_product(out_proj.gguf,yf,scaled_gate,linear_norm,qwen_config.rms_norm_eps,true)
                          :gguf_rms_gated_product(out_proj.gguf,yf,z.reshape({B,T,vsz}),linear_norm,qwen_config.rms_norm_eps,true);
                      if(scaled_output.defined())return scaled_output.values.reshape({B,T,H});
                    }
                    if(scaled_gate.defined())z=gguf_prefill_materialize(scaled_gate).reshape({B,T,vsz});
                    return out_proj.forward_input_rms_silu(
                        execution, yf, z.reshape({B, T, vsz}), linear_norm,
                        qwen_config.rms_norm_eps);
                  });
                }
                z = profiler.measure("linear.z_view", [&]() {
                  return z.reshape({B, T, nv, dv}).transpose(1, 2).contiguous();
                });
                auto zf = profiler.measure("linear.zf_view", [&]() {
                  return z.transpose(1, 2).contiguous().reshape({B, T, vsz});
                });
                return profiler.measure("linear.out_proj_gate_dense", [&]() {
                    auto dtype = out_proj_dense.scalar_type();
                    auto gated =
                        yf.to(dtype) * mfq_tensor_backend::silu(zf.to(dtype));
                    return dense_projection(gated.reshape({B * T, vsz}),
                                            out_proj_dense)
                        .reshape({B, T, H});
                });
              });
        },
        [&](mfq_tensor_backend::Tensor residual,
            mfq_tensor_backend::Tensor oo) {
          auto attn_pair =
              profiler.measure("linear.attn_residual_ffn_norm", [&]() {
                auto rr = residual.reshape({-1, H});
                auto oo2 = oo.reshape({-1, H});
                if(scaled_output.defined())return gguf_prefill_acc_rms(scaled_output,rr,ffn_norm,
                    qwen_config.rms_norm_eps,qwen_config.legacy_tensor_layout.norm_weight_offset);
                if(normalized_ffn_input && !execution.config.diagnostic_fp32_residual && ffn.supports_prepared_gguf_input()) {
                  mfq_tensor_backend::Tensor sum,norm;
                  *normalized_ffn_input=ffn.gate_up.prepare_acc_rms_norm_input(execution,rr,oo2,ffn_norm,
                      qwen_config.rms_norm_eps,qwen_config.legacy_tensor_layout.norm_weight_offset,sum,norm);
                  if(sum.defined())return std::vector<mfq_tensor_backend::Tensor>{sum,norm};
                }

                const bool fused_projection_conversion =
                    !execution.config.diagnostic_fp32_residual &&
                    rr.scalar_type() == mfq_tensor_backend::kFloat16 &&
                    oo2.scalar_type() == mfq_tensor_backend::kFloat32;
                if (oo2.scalar_type() != rr.scalar_type() && !fused_projection_conversion) {
                  oo2 = oo2.to(rr.scalar_type()).contiguous();
                }
                if (execution.config.diagnostic_fp32_residual) {
                  return acc_rms_norm_cuda(
                      rr.to(mfq_tensor_backend::kFloat32),
                      oo2.to(mfq_tensor_backend::kFloat32), ffn_norm,
                      qwen_config.rms_norm_eps,
                      qwen_config.legacy_tensor_layout.norm_weight_offset);
                }
                if (rr.scalar_type() == mfq_tensor_backend::kFloat16 &&
                    (oo2.scalar_type() == mfq_tensor_backend::kFloat16 || fused_projection_conversion)) {
                  return acc_rms_norm_f16_cuda(
                      rr, oo2, ffn_norm, qwen_config.rms_norm_eps,
                      qwen_config.legacy_tensor_layout.norm_weight_offset);
                }
                return acc_rms_norm_cuda(
                    rr, oo2, ffn_norm, qwen_config.rms_norm_eps,
                    qwen_config.legacy_tensor_layout.norm_weight_offset);
              });
          return std::array<mfq_tensor_backend::Tensor, 2>{
              attn_pair[0].reshape({B, T, H}), attn_pair[1].reshape({B, T, H})};
        });
  }

  void replay_recurrent_cuda(const LinearRecurrentInputs &inputs,
                             int64_t retained_tokens,
                             const mfq_tensor_backend::Tensor& initial_convolution={},
                             const mfq_tensor_backend::Tensor& initial_recurrent={}) {
    TYLOINFER_RUNTIME_CHECK(retained_tokens > 0 && retained_tokens <= inputs.tokens &&
                          inputs.batch > 0 && inputs.split == split_in_proj,
                      "invalid linear-attention recurrent replay");
    const int64_t B = inputs.batch;
    const int64_t T = retained_tokens;
    const int64_t nk = qwen_config.linear_num_key_heads;
    const int64_t nv = qwen_config.linear_num_value_heads;
    const int64_t dk = qwen_config.linear_key_head_dim;
    const int64_t dv = qwen_config.linear_value_head_dim;
    const int64_t ksz = qwen_config.linear_k_size();
    const int64_t vsz = qwen_config.linear_v_size();
    if(inputs.fused) {
      TYLOINFER_RUNTIME_CHECK(try_gdn_replay_updates_cuda(inputs.qkv,inputs.value,
        initial_convolution.defined()?initial_convolution:mixer.convolution,
        initial_recurrent.defined()?initial_recurrent:mixer.recurrent,
        mixer.convolution,mixer.recurrent,T,transposed_gdn_state),
        "invalid fused GDN verification update records");
      return;
    }
    if(initial_convolution.defined() && initial_recurrent.defined()) {
      if(!inputs.split && try_gdn_replay_state_cuda(inputs.normalized_k,inputs.convolved_v,
          inputs.gate,inputs.beta,inputs.qkv,initial_convolution,initial_recurrent,
          mixer.convolution,mixer.recurrent,T,transposed_gdn_state,tiled_v_heads))return;
      mixer.convolution.copy_(initial_convolution);
      if(mixer.recurrent.data_ptr()!=initial_recurrent.data_ptr())
        mixer.recurrent.copy_(initial_recurrent);
    }
    auto bias = conv_bias.defined()
                    ? conv_bias
                    : mfq_tensor_backend::empty(
                          {0}, mfq_tensor_backend::TensorOptions()
                                   .device(mfq_tensor_backend::kCUDA)
                                   .dtype(mfq_tensor_backend::kFloat32));
    auto qkv =
        inputs.split
            ? mfq_tensor_backend::cat(
                  {inputs.qk.narrow(1, 0, T).to(mfq_tensor_backend::kFloat32),
                   inputs.value.narrow(1, 0, T).to(
                       mfq_tensor_backend::kFloat32)},
                  -1)
            : inputs.qkv.narrow(1, 0, T);
    auto conv_input = mfq_tensor_backend::cat({mixer.convolution, qkv}, 1);
    mixer.convolution.copy_(conv_input.narrow(
        1, conv_input.size(1) - (qwen_config.linear_conv_kernel_dim - 1),
        qwen_config.linear_conv_kernel_dim - 1));
    mfq_tensor_backend::Tensor q,k,v;
    if(inputs.normalized_q.defined() && inputs.normalized_k.defined() && inputs.convolved_v.defined()) {
      q=inputs.normalized_q.narrow(2,0,T);
      k=inputs.normalized_k.narrow(2,0,T);
      v=inputs.convolved_v.narrow(2,0,T);
    } else {
      auto conv = ssm_conv_silu_cuda(conv_input, conv_weight, bias, T);
      q = conv.narrow(2, 0, ksz).reshape({B, T, nk, dk}).transpose(1, 2);
      k = conv.narrow(2, ksz, ksz).reshape({B, T, nk, dk}).transpose(1, 2);
      v = conv.narrow(2, 2 * ksz, vsz).reshape({B, T, nv, dv}).transpose(1, 2);
      q = l2_norm_cuda(q.contiguous().reshape({-1, dk}), qwen_config.rms_norm_eps)
              .reshape_as(q);
      k = l2_norm_cuda(k.contiguous().reshape({-1, dk}), qwen_config.rms_norm_eps)
              .reshape_as(k);
    }
    const bool transposed = transposed_gdn_state;
    const auto gate = inputs.gate.narrow(2, 0, T).contiguous();
    const auto beta = inputs.beta.narrow(2, 0, T).contiguous();
    std::vector<mfq_tensor_backend::Tensor> output;
    if (transposed) {
      output = tiled_v_heads
                   ? gdn_inplace_transposed_tiled_cuda(
                         q.contiguous(), k.contiguous(), v.contiguous(), gate,
                         beta, mixer.recurrent)
                   : gdn_inplace_transposed_cuda(q.contiguous(), k.contiguous(),
                                                 v.contiguous(), gate, beta,
                                                 mixer.recurrent);
    } else {
      output =
          tiled_v_heads
              ? gdn_inplace_tiled_cuda(q.contiguous(), k.contiguous(),
                                       v.contiguous(), gate, beta, mixer.recurrent)
              : gdn_inplace_cuda(q.contiguous(), k.contiguous(), v.contiguous(),
                                 gate, beta, mixer.recurrent);
    }
    mixer.recurrent = output[1];
  }

  mfq_tensor_backend::Tensor
  forward_ffn_cuda(CudaExecutionContext &execution,
                   mfq_tensor_backend::Tensor residual,
                   mfq_tensor_backend::Tensor xn,
                   const GgufProjectionInput* normalized_ffn_input=nullptr) {
    auto &profiler = execution.profiler;
    const int64_t B = residual.size(0), T = residual.size(1),
                  H = residual.size(2);
    return mfq::models::feed_forward(
        std::move(residual), std::move(xn),
        [&](mfq_tensor_backend::Tensor xn, mfq_tensor_backend::Tensor residual)
            -> std::optional<mfq_tensor_backend::Tensor> {
          auto ffn_input = xn.reshape({B * T, H});
          auto residual_flat = residual.reshape({B * T, H});
          if(auto fused=ffn.try_forward_gguf_prefill_residual(execution,ffn_input,residual_flat))
            return fused->reshape({B,T,H});
          if(normalized_ffn_input) {
            if(auto fused=ffn.try_forward_prepared_gguf_residual(
                execution,*normalized_ffn_input,residual_flat))return fused->reshape({B,T,H});
          }
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
          return (normalized_ffn_input && !normalized_ffn_input->shape.empty()
                      ?ffn.forward_prepared_gguf(execution,*normalized_ffn_input)
                      :ffn.forward(execution,xn.reshape({B*T,H})))
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

  mfq_tensor_backend::Tensor
  forward(CudaExecutionContext &execution, mfq_tensor_backend::Tensor x,
          mfq_tensor_backend::Tensor pos, int64_t cache_pos,
          const MfqOptional<mfq_tensor_backend::Tensor> &seq_len,
          const RopeCache &rope,
          const MfqOptional<mfq_tensor_backend::Tensor> &cache_positions =
              mfq_nullopt,
          const MfqOptional<mfq_tensor_backend::Tensor> &attention_mask =
              mfq_nullopt) override {
    (void)cache_positions;
    (void)attention_mask;
    if (!x.is_cuda()) {
      return forward_cpu(execution, std::move(x));
    }
    (void)seq_len;
    (void)pos;
    (void)cache_pos;
    (void)rope;
    GgufProjectionInput normalized_ffn_input;
    auto attention=forward_attention_cuda(execution,std::move(x),nullptr,&normalized_ffn_input);
    return forward_ffn_cuda(execution,std::move(attention[0]),std::move(attention[1]),&normalized_ffn_input);
  }
  mfq_tensor_backend::Tensor decode_core_workspace;

};

} // namespace mfq::cuda::qwen35
