#include "linear.h"
#include "engine/ops.h"
#include "engine/backends/cuda/ops/attention/attention.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/kernels/mfq_cuda_linear_attention_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/weights/group.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

using ::tyloi::indexing::Slice;

namespace tyloi::cuda {

namespace {

template <class DenseTensor> struct LinearProjections {
    DenseTensor qkv, qk, value, output_gate, alpha, beta;
};

// Convolution/QK normalization can be one fused native operator. GDN updates
// its native recurrent state; the model owns its placement in the graph.
template <class DenseTensor, class Project, class Gates, class Convolve, class Recur, class Normalize,
          class Output>
DenseTensor linear_attention(DenseTensor normalized, Project project, Gates gates, Convolve convolve,
                        Recur recur, Normalize normalize, Output output) {
    auto projected = project(std::move(normalized));
    auto recurrence_gates = gates(projected);
    auto qkv = convolve(projected);
    auto hidden = recur(std::move(qkv), std::move(recurrence_gates));
    auto normed = normalize(std::move(hidden));
    return output(std::move(normed), std::move(projected.output_gate));
}

} // namespace

std::array<::tyloi::DenseTensor, 2> LinearAttention::forward(
        const Projector& project, CudaProfiler& profiler, const Options& options,
        ::tyloi::DenseTensor input,
        LinearView state, LinearInputs* captured) const {
    TYLOINFER_RUNTIME_CHECK(dense_ab_tail || !split_in_proj || split_dense_zab || z_proj,
        "linear attention Z projection is missing");
    TYLOINFER_RUNTIME_CHECK(dense_out_proj || out_proj,
        "linear attention output projection is missing");
    return input.is_cuda() ? cuda(project, profiler, options, std::move(input), state, captured)
                           : cpu(project, std::move(input), state);
}

std::array<::tyloi::DenseTensor, 2> LinearAttention::cpu(
        const Projector& project, ::tyloi::DenseTensor x,
        LinearView state) const {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    const auto in_group = !split_in_proj ? project.group(in_proj) : QuantLinearGroup{};
    const auto qkv_group = split_in_proj ? project.group(qkv_proj) : QuantLinearGroup{};
    const auto ab_group = split_in_proj && !split_dense_zab && ab_is_nint
        ? project.group(ab_nint_proj) : QuantLinearGroup{};
#else
    const auto in_group = !split_in_proj ? make_quant_group(in_proj, grouped_weights, true) : QuantLinearGroup{};
    const auto qkv_group = split_in_proj ? make_quant_group(qkv_proj, grouped_weights, true) : QuantLinearGroup{};
    const auto ab_group = split_in_proj && !split_dense_zab && ab_is_nint
        ? make_quant_group(ab_nint_proj, grouped_weights, true) : QuantLinearGroup{};
#endif
    const int64_t B = x.size(0), T = x.size(1), H = x.size(2);
    const int64_t nk = key_heads;
    const int64_t nv = value_heads;
    const int64_t dk = key_dim;
    const int64_t dv = value_dim;
    const int64_t ksz = (key_heads * key_dim);
    const int64_t vsz = (value_heads * value_dim);
    const int64_t conv_dim = 2 * ksz + vsz;

    auto xn = qwen_rms_norm(
                  x.reshape({B * T, H}).to(::tyloi::kFloat32),
                  attn_norm, rms_norm_eps, norm_weight_offset)
                  .reshape({B, T, H});
    auto output = linear_attention(
              std::move(xn),
              [&](::tyloi::DenseTensor xn) {
                ::tyloi::DenseTensor qkv, qk_part, v_part, z, alpha_raw,
                    beta_raw;
                if (dense_ab_tail) {
                  auto parts = project.run(in_group, xn);
                  qkv = parts[0].to(::tyloi::kFloat32);
                  z = parts[1];
                  auto ab = ab_proj.forward(xn);
                  alpha_raw = ab[0];
                  beta_raw = ab[1];
                } else if (split_in_proj) {
                  if (split_dense_zab) {
                    auto qkv_parts = project.run(qkv_group, xn);
                    qk_part = qkv_parts[0];
                    v_part = qkv_parts[1];
                    auto zab = zab_proj.forward(xn);
                    z = zab[0];
                    alpha_raw = zab[1];
                    beta_raw = zab[2];
                  } else {
                    auto qkv_parts = project.run(qkv_group, xn);
                    qk_part = qkv_parts[0];
                    v_part = qkv_parts[1];
                    z = project.run(*z_proj, xn);
                    auto ab = ab_is_nint ? project.run(ab_group, xn)
                                         : ab_proj.forward(xn);
                    alpha_raw = ab[0];
                    beta_raw = ab[1];
                  }
                  qkv = ::tyloi::cuda::cat(
                      {qk_part.to(::tyloi::kFloat32),
                       v_part.to(::tyloi::kFloat32)},
                      -1);
                } else {
                  auto parts = project.run(in_group, xn);
                  qkv = parts[0].to(::tyloi::kFloat32);
                  z = parts[1];
                  alpha_raw = parts[2];
                  beta_raw = parts[3];
                }

                return LinearProjections<
                    ::tyloi::DenseTensor>{qkv, qk_part,   v_part,
                                                z,   alpha_raw, beta_raw};
              },
              [&](const LinearProjections<
                  ::tyloi::DenseTensor> &p) {
                auto alpha_raw = p.alpha, beta_raw = p.beta;
                auto beta = ::tyloi::cuda::sigmoid(
                    beta_raw.to(::tyloi::kFloat32)
                        .reshape({B, T, nv}));
                auto alpha = alpha_raw.to(::tyloi::kFloat32)
                                 .reshape({B, T, nv});
                auto gate = ::tyloi::cuda::softplus(
                    alpha + dt_bias.to(::tyloi::kFloat32)
                                .reshape({1, 1, nv}));
                auto coefficient = -::tyloi::cuda::exp(
                    a_log.to(::tyloi::kFloat32));
                gate = gate * coefficient.reshape({1, 1, nv});

                auto gate_t = gate.transpose(1, 2).contiguous();
                auto beta_t = beta.transpose(1, 2).contiguous();

                return std::array<::tyloi::DenseTensor, 2>{gate_t,
                                                                 beta_t};
              },
              [&](const LinearProjections<
                  ::tyloi::DenseTensor> &p) {
                auto qkv = p.qkv, qk_part = p.qk, v_part = p.value;
                auto conv_input = ::tyloi::cuda::cat({state.convolution, qkv}, 1);
                state.convolution.copy_(conv_input.narrow(
                    1,
                    conv_input.size(1) -
                        (conv_kernel - 1),
                    conv_kernel - 1));
                auto weight = conv_weight.to(::tyloi::kFloat32);
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
                                          conv_kernel,
                                  "CPU SSM convolution geometry mismatch");
                auto windows =
                    conv_input.unfold(1, conv_kernel, 1);
                auto conv =
                    (windows *
                     weight.reshape(
                         {1, 1, conv_dim, conv_kernel}))
                        .sum(-1);
                if (conv_bias.defined()) {
                  conv = conv + conv_bias.to(::tyloi::kFloat32)
                                    .reshape({1, 1, conv_dim});
                }
                conv = ::tyloi::cuda::silu(conv);

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
                q = q / ::tyloi::cuda::sqrt(q.square().sum(-1, true))
                            .clamp_min(rms_norm_eps);
                k = k / ::tyloi::cuda::sqrt(k.square().sum(-1, true))
                            .clamp_min(rms_norm_eps);
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

                return std::array<::tyloi::DenseTensor, 3>{q, k, v};
              },
              [&](std::array<::tyloi::DenseTensor, 3> qkv,
                  std::array<::tyloi::DenseTensor, 2> gates) {
                auto q = qkv[0], k = qkv[1], v = qkv[2];
                auto gate_t = gates[0], beta_t = gates[1];
                auto recurrent = state.recurrent;
                std::vector<::tyloi::DenseTensor> outputs;
                outputs.reserve(static_cast<size_t>(T));
                const double retrieve_scale =
                    1.0 / std::sqrt(static_cast<double>(dk));
                for (int64_t token = 0; token < T; ++token) {
                  auto qt = q.select(2, token);
                  auto kt = k.select(2, token);
                  auto vt = v.select(2, token);
                  recurrent =
                      recurrent * ::tyloi::cuda::exp(gate_t.select(2, token))
                                  .reshape({B, nv, 1, 1});
                  auto state_k = ::tyloi::cuda::matmul(
                                     recurrent.transpose(-1, -2), kt.unsqueeze(-1))
                                     .squeeze(-1);
                  auto delta =
                      (vt - state_k) * beta_t.select(2, token).unsqueeze(-1);
                  recurrent = recurrent + kt.unsqueeze(-1) * delta.unsqueeze(-2);
                  outputs.push_back(
                      ::tyloi::cuda::matmul(recurrent.transpose(-1, -2),
                                                 qt.unsqueeze(-1))
                          .squeeze(-1) *
                      retrieve_scale);
                }
                state.recurrent = recurrent.contiguous();
                auto y = ::tyloi::cuda::stack(outputs, 2);

                return y;
              },
              [&](::tyloi::DenseTensor y) {
                auto y_flat =
                    y.reshape({-1, dv}).to(::tyloi::kFloat32);
                auto inverse = ::tyloi::cuda::rsqrt(
                    y_flat.square().mean(-1, true) + rms_norm_eps);
                auto y_norm = (y_flat * inverse *
                               linear_norm.to(::tyloi::kFloat32))
                                  .reshape_as(y);
                auto yf =
                    y_norm.transpose(1, 2).contiguous().reshape({B, T, vsz});

                return yf;
              },
              [&](::tyloi::DenseTensor yf, ::tyloi::DenseTensor z) {
                z = z.reshape({B, T, nv, dv}).transpose(1, 2).contiguous();
                auto zf = z.transpose(1, 2).contiguous().reshape({B, T, vsz});
                ::tyloi::DenseTensor attention_output;
                if (dense_out_proj) {
                  auto dtype = out_proj_dense.scalar_type();
                  auto gated =
                      yf.to(dtype) * ::tyloi::cuda::silu(zf.to(dtype));
                  attention_output =
                      dense_projection(gated.reshape({B * T, vsz}),
                                       out_proj_dense)
                          .reshape({B, T, H});
                } else {
                  attention_output =
                      project.gated(*out_proj, yf, zf, InputGateMode::Silu);
                }
                return attention_output;
              });
    return std::array<::tyloi::DenseTensor, 2>{std::move(x), std::move(output)};
}

std::array<::tyloi::DenseTensor, 2> LinearAttention::cuda(
        const Projector& project, CudaProfiler& profiler, const Options& options,
        ::tyloi::DenseTensor x,
        LinearView state, LinearInputs* captured) const {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    const auto in_group = !split_in_proj ? project.group(in_proj) : QuantLinearGroup{};
    const auto qkv_group = split_in_proj ? project.group(qkv_proj) : QuantLinearGroup{};
    const auto ab_group = split_in_proj && !split_dense_zab && ab_is_nint
        ? project.group(ab_nint_proj) : QuantLinearGroup{};
#else
    const auto in_group = !split_in_proj ? make_quant_group(in_proj, grouped_weights, false) : QuantLinearGroup{};
    const auto qkv_group = split_in_proj ? make_quant_group(qkv_proj, grouped_weights, false) : QuantLinearGroup{};
    const auto ab_group = split_in_proj && !split_dense_zab && ab_is_nint
        ? make_quant_group(ab_nint_proj, grouped_weights, false) : QuantLinearGroup{};
#endif
    int64_t B = x.size(0), T = x.size(1), H = x.size(2);
    int64_t nk = key_heads,
            nv = value_heads;
    int64_t dk = key_dim,
            dv = value_dim;
    int64_t ksz = (key_heads * key_dim),
            vsz = (value_heads * value_dim);

    auto xn = profiler.measure("linear.attn_norm", [&]() {
            return qwen_rms_norm(
                       x.reshape({B * T, H}).to(::tyloi::kFloat32),
                       attn_norm, rms_norm_eps,
                       norm_weight_offset)
                .reshape({B, T, H});
          });
    auto output = linear_attention(
              std::move(xn),
              [&](::tyloi::DenseTensor xn) {
                ::tyloi::DenseTensor qkv, qk_part, v_part, z, alpha_raw,
                    beta_raw;
                if (dense_ab_tail) {
                  auto parts = profiler.measure("linear.in_proj", [&]() {
                    return project.run(in_group, xn);
                  });
                  qkv = profiler.measure("linear.qkv_cast", [&]() {
                    return parts[0].to(::tyloi::kFloat32);
                  });
                  z = parts[1];
                  auto ab = profiler.measure(
                      "linear.ab_proj", [&]() { return ab_proj.forward(xn); });
                  alpha_raw = ab[0];
                  beta_raw = ab[1];
                } else if (split_in_proj) {
                  bool shared_projection_input = false;
                  if (!shared_projection_input && !split_dense_zab &&
                      ab_is_nint &&
                      options.shared_input &&
                      qkv_proj.size() == 2) {
                    std::vector<const ::tyloi::Tensor*> projections = {
                        qkv_proj[0].get(),
                        qkv_proj[1].get(),
                        z_proj.get(),
                    };
                    const bool shared_tp_ab =
                        ab_nint_proj.size() == 2 &&
                        dynamic_cast<const ::tyloi::ShardedTensor*>(ab_nint_proj[0].get()) &&
                        dynamic_cast<const ::tyloi::ShardedTensor*>(ab_nint_proj[1].get());
                    if (shared_tp_ab) {
                      projections.push_back(ab_nint_proj[0].get());
                      projections.push_back(ab_nint_proj[1].get());
                    }
                    if (tensor_parallel_output_projections_compatible(
                            projections)) {
                      auto parts =
                          profiler.measure("linear.split_projections", [&]() {
                            return project.grouped(projections, xn);
                          });
                      qk_part = parts[0];
                      v_part = parts[1];
                      z = parts[2];
                      if (shared_tp_ab) {
                        alpha_raw = parts[3];
                        beta_raw = parts[4];
                      } else {
                        auto ab = profiler.measure("linear.ab_proj", [&]() {
                          return project.run(ab_group, xn);
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
                            return project.run(qkv_group, xn);
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
                            return project.run(qkv_group, xn);
                          });
                      qk_part = qkv_parts[0];
                      v_part = qkv_parts[1];
                      z = profiler.measure("linear.z_proj", [&]() {
                        return project.run(*z_proj, xn);
                      });
                      auto ab = profiler.measure("linear.ab_proj", [&]() {
                        return ab_is_nint ? project.run(ab_group, xn)
                                          : ab_proj.forward(xn);
                      });
                      alpha_raw = ab[0];
                      beta_raw = ab[1];
                    }
                  }
                } else {
                  auto parts = profiler.measure("linear.in_proj", [&]() {
                    return project.run(in_group, xn);
                  });
                  qkv = profiler.measure("linear.qkv_cast", [&]() {
                    return parts[0].to(::tyloi::kFloat32);
                  });
                  z = parts[1];
                  alpha_raw = parts[2];
                  beta_raw = parts[3];
                }

                return LinearProjections<
                    ::tyloi::DenseTensor>{qkv, qk_part,   v_part,
                                                z,   alpha_raw, beta_raw};
              },
              [&](const LinearProjections<
                  ::tyloi::DenseTensor> &p) {
                auto qkv = p.qkv, qk_part = p.qk, v_part = p.value;
                auto alpha_raw = p.alpha, beta_raw = p.beta;
                auto gates = profiler.measure("linear.gates_fused", [&]() {
                  return linear_gate_beta_cuda(alpha_raw.reshape({B, T, nv}),
                                               beta_raw.reshape({B, T, nv}),
                                               dt_bias, a_log);
                });
                auto gate_t = gates[0];
                auto beta_t = gates[1];
                if (captured != nullptr) {
                  captured->qk = qk_part;
                  captured->value = v_part;
                  captured->qkv = qkv;
                  captured->gate = gate_t;
                  captured->beta = beta_t;
                  captured->batch = B;
                  captured->tokens = T;
                  captured->split = split_in_proj;
                }

                return std::array<::tyloi::DenseTensor, 2>{gate_t,
                                                                 beta_t};
              },
              [&](const LinearProjections<
                  ::tyloi::DenseTensor> &p) {
                auto qkv = p.qkv, qk_part = p.qk, v_part = p.value;
                auto bias =
                    conv_bias.defined()
                        ? conv_bias
                        : ::tyloi::cuda::empty(
                              {0}, ::tyloi::TensorOptions()
                                       .device(::tyloi::kCUDA)
                                       .dtype(::tyloi::kFloat32));
                ::tyloi::DenseTensor q, k, v;
                bool fused_prefill =
                    T >= 256 && options.fused_conv;
                if (T > 1 && split_in_proj && fused_prefill) {
                  auto qkv_fast =
                      profiler.measure("linear.conv_qkv_prefill", [&]() {
                        return linear_conv_qkv_prefill_cuda(
                            state.convolution, qk_part.contiguous(),
                            v_part.contiguous(), conv_weight, bias, nk, nv, dk,
                            dv, rms_norm_eps);
                      });
                  q = qkv_fast[0];
                  k = qkv_fast[1];
                  v = qkv_fast[2];
                  // Runtime CUDA graphs retain this state storage address
                  // across requests.
                  state.convolution.copy_(qkv_fast[3]);
                } else if (T == 1 && split_in_proj) {
                  auto qkv_fast =
                      profiler.measure("linear.conv_qkv_decode", [&]() {
                        return linear_conv_qkv_decode_cuda(
                            state.convolution, qk_part.contiguous(),
                            v_part.contiguous(), conv_weight, bias, nk, nv, dk,
                            dv, rms_norm_eps);
                      });
                  q = qkv_fast[0];
                  k = qkv_fast[1];
                  v = qkv_fast[2];
                } else {
                  if (split_in_proj) {
                    qkv = profiler.measure("linear.qkv_cat", [&]() {
                      return ::tyloi::cuda::cat(
                          {qk_part.to(::tyloi::kFloat32),
                           v_part.to(::tyloi::kFloat32)},
                          -1);
                    });
                  }
                  ::tyloi::DenseTensor conv;
                  if (T == 1) {
                    conv = profiler.measure("linear.conv_silu_decode", [&]() {
                      return ssm_conv_silu_decode_cuda(state.convolution, qkv,
                                                       conv_weight, bias);
                    });
                  } else {
                    auto conv_in = profiler.measure("linear.conv_input", [&]() {
                      return ::tyloi::cuda::cat({state.convolution, qkv}, 1);
                    });
                    auto next_conv_state =
                        profiler.measure("linear.conv_state_update", [&]() {
                          return conv_in
                              .index(
                                  {Slice(),
                                   Slice(-(conv_kernel -
                                           1),
                                         ::tyloi::indexing::None),
                                   Slice()})
                              .contiguous();
                        });
                    state.convolution.copy_(next_conv_state);
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
                                        rms_norm_eps)
                        .reshape_as(q);
                  });
                  k = profiler.measure("linear.k_l2", [&]() {
                    return l2_norm_cuda(k.contiguous().reshape({-1, dk}),
                                        rms_norm_eps)
                        .reshape_as(k);
                  });
                }

                return std::array<::tyloi::DenseTensor, 3>{q, k, v};
              },
              [&](std::array<::tyloi::DenseTensor, 3> qkv,
                  std::array<::tyloi::DenseTensor, 2> gates) {
                auto q = qkv[0], k = qkv[1], v = qkv[2];
                auto gate_t = gates[0], beta_t = gates[1];
                auto recurrent_step = [&](::tyloi::DenseTensor rq,
                                          ::tyloi::DenseTensor rk,
                                          ::tyloi::DenseTensor rv,
                                          ::tyloi::DenseTensor rg,
                                          ::tyloi::DenseTensor rb) {
                  const bool transposed = state.transposed;
                  if (transposed) {
                    if (tiled_v_heads)
                      return gdn_inplace_transposed_tiled_cuda(
                          rq.contiguous(), rk.contiguous(), rv.contiguous(), rg,
                          rb, state.recurrent);
                    return gdn_inplace_transposed_cuda(
                        rq.contiguous(), rk.contiguous(), rv.contiguous(), rg,
                        rb, state.recurrent);
                  }
                  if (tiled_v_heads)
                    return gdn_inplace_tiled_cuda(
                        rq.contiguous(), rk.contiguous(), rv.contiguous(), rg,
                        rb, state.recurrent);
                  return gdn_inplace_cuda(rq.contiguous(), rk.contiguous(),
                                          rv.contiguous(), rg, rb, state.recurrent);
                };
                auto gd = profiler.measure("linear.gdn", [&]() {
                  return recurrent_step(q, k, v, gate_t, beta_t);
                });
                auto y = gd[0];
                state.recurrent = gd[1];

                return y;
              },
              [&](::tyloi::DenseTensor y) {
                auto y_norm = profiler.measure("linear.out_norm", [&]() {
                  return rms_norm_cuda(y.reshape({-1, dv}).to(
                                           ::tyloi::kFloat32),
                                       linear_norm, rms_norm_eps)
                      .reshape_as(y);
                });
                auto yf = profiler.measure("linear.y_view", [&]() {
                  return y_norm.transpose(1, 2).contiguous().reshape(
                      {B, T, vsz});
                });

                return yf;
              },
              [&](::tyloi::DenseTensor yf, ::tyloi::DenseTensor z) {
                ::tyloi::DenseTensor oo;
                z = profiler.measure("linear.z_view", [&]() {
                  return z.reshape({B, T, nv, dv}).transpose(1, 2).contiguous();
                });
                auto zf = profiler.measure("linear.zf_view", [&]() {
                  return z.transpose(1, 2).contiguous().reshape({B, T, vsz});
                });
                if (dense_out_proj) {
                  oo = profiler.measure("linear.out_proj_gate_dense", [&]() {
                    auto dtype = out_proj_dense.scalar_type();
                    auto gated =
                        yf.to(dtype) * ::tyloi::cuda::silu(zf.to(dtype));
                    return dense_projection(gated.reshape({B * T, vsz}),
                                            out_proj_dense)
                        .reshape({B, T, H});
                  });
                } else {
                  oo = profiler.measure("linear.out_proj_gate", [&]() {
                    return project.gated(*out_proj, yf, zf, InputGateMode::Silu);
                  });
                }
                return oo;
              });
    return std::array<::tyloi::DenseTensor, 2>{std::move(x), std::move(output)};
}

void LinearAttention::replay(
        LinearView state, const LinearInputs& inputs, int64_t retained_tokens) const {
    TYLOINFER_RUNTIME_CHECK(retained_tokens > 0 && retained_tokens <= inputs.tokens &&
                          inputs.batch > 0 && inputs.split == split_in_proj,
                      "invalid linear-attention recurrent replay");
    const int64_t B = inputs.batch;
    const int64_t T = retained_tokens;
    const int64_t nk = key_heads;
    const int64_t nv = value_heads;
    const int64_t dk = key_dim;
    const int64_t dv = value_dim;
    const int64_t ksz = (key_heads * key_dim);
    const int64_t vsz = (value_heads * value_dim);
    auto bias = conv_bias.defined()
                    ? conv_bias
                    : ::tyloi::cuda::empty(
                          {0}, ::tyloi::TensorOptions()
                                   .device(::tyloi::kCUDA)
                                   .dtype(::tyloi::kFloat32));
    auto qkv =
        inputs.split
            ? ::tyloi::cuda::cat(
                  {inputs.qk.narrow(1, 0, T).to(::tyloi::kFloat32),
                   inputs.value.narrow(1, 0, T).to(
                       ::tyloi::kFloat32)},
                  -1)
            : inputs.qkv.narrow(1, 0, T);
    auto conv_input = ::tyloi::cuda::cat({state.convolution, qkv}, 1);
    state.convolution.copy_(conv_input.narrow(
        1, conv_input.size(1) - (conv_kernel - 1),
        conv_kernel - 1));
    auto conv = ssm_conv_silu_cuda(conv_input, conv_weight, bias, T);
    auto q = conv.narrow(2, 0, ksz).reshape({B, T, nk, dk}).transpose(1, 2);
    auto k = conv.narrow(2, ksz, ksz).reshape({B, T, nk, dk}).transpose(1, 2);
    auto v =
        conv.narrow(2, 2 * ksz, vsz).reshape({B, T, nv, dv}).transpose(1, 2);
    q = l2_norm_cuda(q.contiguous().reshape({-1, dk}), rms_norm_eps)
            .reshape_as(q);
    k = l2_norm_cuda(k.contiguous().reshape({-1, dk}), rms_norm_eps)
            .reshape_as(k);
    const bool transposed = state.transposed;
    const auto gate = inputs.gate.narrow(2, 0, T).contiguous();
    const auto beta = inputs.beta.narrow(2, 0, T).contiguous();
    std::vector<::tyloi::DenseTensor> output;
    if (transposed) {
      output = tiled_v_heads
                   ? gdn_inplace_transposed_tiled_cuda(
                         q.contiguous(), k.contiguous(), v.contiguous(), gate,
                         beta, state.recurrent)
                   : gdn_inplace_transposed_cuda(q.contiguous(), k.contiguous(),
                                                 v.contiguous(), gate, beta,
                                                 state.recurrent);
    } else {
      output =
          tiled_v_heads
              ? gdn_inplace_tiled_cuda(q.contiguous(), k.contiguous(),
                                       v.contiguous(), gate, beta, state.recurrent)
              : gdn_inplace_cuda(q.contiguous(), k.contiguous(), v.contiguous(),
                                 gate, beta, state.recurrent);
    }
    state.recurrent = output[1];
}

} // namespace tyloi::cuda
