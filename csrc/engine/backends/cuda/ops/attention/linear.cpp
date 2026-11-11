#include "linear.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/ops/linear.h"
#include "engine/state/recurrent.h"
#include "engine/backends/cuda/executor/executor.h"
#define TYLOI_LINEAR_IMPL LinearCore
#else
#define TYLOI_LINEAR_IMPL LinearAttention
#endif
#include "engine/ops.h"
#include "engine/backends/cuda/ops/attention/attention.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/kernels/mfq_cuda_linear_attention_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/ops/projection/projection.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <limits>
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

std::array<::tyloi::DenseTensor, 2> TYLOI_LINEAR_IMPL::forward(
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

std::array<::tyloi::DenseTensor, 2> TYLOI_LINEAR_IMPL::cpu(
        const Projector& project, ::tyloi::DenseTensor x,
        LinearView state) const {
#ifdef TYLOINFER_NATIVE_CUDA
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

#ifdef TYLOINFER_NATIVE_CUDA
    const auto& attn_norm = this->attn_norm.dense();
#endif
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
                    z = project.run(z_proj, xn);
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
#ifdef TYLOINFER_NATIVE_CUDA
                const auto& dt_bias = this->dt_bias.dense();
#endif
                auto gate = ::tyloi::cuda::softplus(
                    alpha + dt_bias.to(::tyloi::kFloat32)
                                .reshape({1, 1, nv}));
#ifdef TYLOINFER_NATIVE_CUDA
                const auto& a_log = this->a_log.dense();
#endif
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
#ifdef TYLOINFER_NATIVE_CUDA
                const auto& conv_weight = this->conv_weight.dense();
#endif
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
                if (
#ifdef TYLOINFER_NATIVE_CUDA
                    conv_bias && conv_bias.defined()
#else
                    conv_bias.defined()
#endif
                ) {
#ifdef TYLOINFER_NATIVE_CUDA
                  const auto& conv_bias = this->conv_bias.dense();
#endif
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
#ifdef TYLOINFER_NATIVE_CUDA
                const auto& linear_norm = this->linear_norm.dense();
#endif
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
#ifdef TYLOINFER_NATIVE_CUDA
                  const auto& out_proj_dense = this->out_proj_dense.dense();
#endif
                  auto dtype = out_proj_dense.scalar_type();
                  auto gated =
                      yf.to(dtype) * ::tyloi::cuda::silu(zf.to(dtype));
                  attention_output =
                      dense_projection(gated.reshape({B * T, vsz}),
                                       out_proj_dense)
                          .reshape({B, T, H});
                } else {
                  attention_output =
                      project.gated(out_proj, yf, zf, InputGateMode::Silu);
                }
                return attention_output;
              });
    return std::array<::tyloi::DenseTensor, 2>{std::move(x), std::move(output)};
}

std::array<::tyloi::DenseTensor, 2> TYLOI_LINEAR_IMPL::cuda(
        const Projector& project, CudaProfiler& profiler, const Options& options,
        ::tyloi::DenseTensor x,
        LinearView state, LinearInputs* captured) const {
#ifdef TYLOINFER_NATIVE_CUDA
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
#ifdef TYLOINFER_NATIVE_CUDA
            const auto& attn_norm = this->attn_norm.dense();
#endif
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
                    std::vector<::tyloi::Tensor> projections = {
                        qkv_proj[0],
                        qkv_proj[1],
                        z_proj,
                    };
                    const bool shared_tp_ab =
                        ab_nint_proj.size() == 2 &&
                        ab_nint_proj[0].representation<::tyloi::ShardedTensor>() &&
                        ab_nint_proj[1].representation<::tyloi::ShardedTensor>();
                    if (shared_tp_ab) {
                      projections.push_back(ab_nint_proj[0]);
                      projections.push_back(ab_nint_proj[1]);
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
                        return project.run(z_proj, xn);
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
#ifdef TYLOINFER_NATIVE_CUDA
                  const auto& dt_bias = this->dt_bias.dense();
                  const auto& a_log = this->a_log.dense();
#endif
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
#ifdef TYLOINFER_NATIVE_CUDA
                std::optional<::tyloi::DenseTensor> empty_bias;
                const auto& bias = conv_bias && conv_bias.defined()
                        ? conv_bias.dense()
                        : empty_bias.emplace(::tyloi::cuda::empty(
                              {0}, ::tyloi::TensorOptions()
                                       .device(::tyloi::kCUDA)
                                       .dtype(::tyloi::kFloat32)));
                const auto& conv_weight = this->conv_weight.dense();
#else
                auto bias =
                    conv_bias.defined()
                        ? conv_bias
                        : ::tyloi::cuda::empty(
                              {0}, ::tyloi::TensorOptions()
                                       .device(::tyloi::kCUDA)
                                       .dtype(::tyloi::kFloat32));
#endif
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
#ifdef TYLOINFER_NATIVE_CUDA
                  const auto& linear_norm = this->linear_norm.dense();
#endif
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
#ifdef TYLOINFER_NATIVE_CUDA
                    const auto& out_proj_dense = this->out_proj_dense.dense();
#endif
                    auto dtype = out_proj_dense.scalar_type();
                    auto gated =
                        yf.to(dtype) * ::tyloi::cuda::silu(zf.to(dtype));
                    return dense_projection(gated.reshape({B * T, vsz}),
                                            out_proj_dense)
                        .reshape({B, T, H});
                  });
                } else {
                  oo = profiler.measure("linear.out_proj_gate", [&]() {
                    return project.gated(out_proj, yf, zf, InputGateMode::Silu);
                  });
                }
                return oo;
              });
    return std::array<::tyloi::DenseTensor, 2>{std::move(x), std::move(output)};
}

void TYLOI_LINEAR_IMPL::replay(
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
#ifdef TYLOINFER_NATIVE_CUDA
    std::optional<::tyloi::DenseTensor> empty_bias;
    const auto& bias = conv_bias && conv_bias.defined()
                    ? conv_bias.dense()
                    : empty_bias.emplace(::tyloi::cuda::empty(
                          {0}, ::tyloi::TensorOptions()
                                   .device(::tyloi::kCUDA)
                                   .dtype(::tyloi::kFloat32)));
#else
    auto bias = conv_bias.defined()
                    ? conv_bias
                    : ::tyloi::cuda::empty(
                          {0}, ::tyloi::TensorOptions()
                                   .device(::tyloi::kCUDA)
                                   .dtype(::tyloi::kFloat32));
#endif
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
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& conv_weight = this->conv_weight.dense();
#endif
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

#undef TYLOI_LINEAR_IMPL
#ifdef TYLOINFER_NATIVE_CUDA
namespace tyloi::ops {
struct Linear::State::Impl {
    cuda::Recurrent mixer;
    cuda::LinearInputs projected;
};
Linear::State::State() : impl_(std::make_unique<Impl>()) {}
Linear::State::~State() = default;
Linear::State::State(State&&) noexcept = default;
Linear::State& Linear::State::operator=(State&&) noexcept = default;
bool Linear::State::defined() const noexcept { return impl_->mixer.defined(); }
bool Linear::State::pending() const noexcept { return impl_->mixer.pending(); }
bool Linear::State::has_tape() const noexcept {
    const auto& p = impl_->projected;
    return p.qk.defined() || p.value.defined() || p.qkv.defined() || p.gate.defined() ||
        p.beta.defined() || p.batch != 0 || p.tokens != 0 || p.split;
}
Linear::State::Snapshot Linear::State::view() const {
    const auto& v = impl_->mixer.view();
    return {v.convolution, v.recurrent, v.transposed};
}
Linear::State::Snapshot Linear::State::snapshot() const {
    std::optional<MfqCudaGuard> guard;
    if (defined() && impl_->mixer.view().convolution.device().is_cuda()) guard.emplace(impl_->mixer.view().convolution.device());
    const auto v = impl_->mixer.snapshot();
    return {v.convolution, v.recurrent, v.transposed};
}
void Linear::State::restore(const Snapshot& value) {
    if (!value.convolution.defined() || !value.recurrent.defined())
        throw std::runtime_error("recurrent snapshot is unavailable");
    std::optional<MfqCudaGuard> guard;
    if (value.convolution.device().is_cuda()) guard.emplace(value.convolution.device());
    auto& live = impl_->mixer.view();
    const auto prepare = [](Tensor& destination, const Tensor& source) {
        bool same = destination.defined() && destination.dim() == source.dim() &&
            destination.scalar_type() == source.scalar_type() && destination.device() == source.device();
        if (same) for (std::int64_t i = 0; i < source.dim(); ++i)
            if (destination.size(i) != source.size(i)) { same = false; break; }
        if (!same) {
            std::vector<std::int64_t> shape;
            for (std::int64_t i = 0; i < source.dim(); ++i) shape.push_back(source.size(i));
            destination = ops::empty(shape, source.options());
        }
    };
    prepare(live.convolution, value.convolution);
    prepare(live.recurrent, value.recurrent);
    impl_->mixer.restore({value.convolution, value.recurrent, value.transposed});
    impl_->projected = {};
}
void Linear::State::initialize(const Snapshot& layout, std::int64_t batch) {
    if (defined() || !layout.convolution.defined() || !layout.recurrent.defined() || batch <= 0)
        throw std::runtime_error("invalid recurrent batch initialization");
    std::optional<MfqCudaGuard> guard;
    if (layout.convolution.device().is_cuda()) guard.emplace(layout.convolution.device());
    auto shape = [batch](const Tensor& value) {
        std::vector<std::int64_t> result;
        for (std::int64_t axis = 0; axis < value.dim(); ++axis) result.push_back(value.size(axis));
        if (result.empty()) throw std::runtime_error("invalid recurrent batch geometry");
        result[0] = batch;
        return result;
    };
    auto& live = impl_->mixer.view();
    live.convolution = ops::zeros(shape(layout.convolution), layout.convolution.options());
    live.recurrent = ops::zeros(shape(layout.recurrent), layout.recurrent.options());
    live.transposed = layout.transposed;
}
void Linear::State::reset(std::int64_t batch) {
    std::optional<MfqCudaGuard> guard;
    if (defined() && impl_->mixer.view().convolution.device().is_cuda()) guard.emplace(impl_->mixer.view().convolution.device());
    impl_->mixer.reset(batch); impl_->projected = {};
}
void Linear::State::commit() noexcept { impl_->projected = {}; impl_->mixer.commit(); }
std::int64_t Linear::State::rewind(std::int64_t position) {
    std::optional<MfqCudaGuard> guard;
    if (defined() && impl_->mixer.view().convolution.device().is_cuda()) guard.emplace(impl_->mixer.view().convolution.device());
    return impl_->mixer.rewind(position);
}
void Linear::State::abort() {
    std::optional<MfqCudaGuard> guard;
    if (defined() && impl_->mixer.view().convolution.device().is_cuda()) guard.emplace(impl_->mixer.view().convolution.device());
    try { impl_->mixer.abort(); }
    catch (...) { impl_->projected = {}; throw; }
    impl_->projected = {};
}
std::vector<Tensor*> Linear::State::graph_state() {
    if (!defined() || pending() || has_tape()) throw std::runtime_error("decode graph requires confirmed recurrent state");
    auto& v = impl_->mixer.view();
    return {&v.convolution, &v.recurrent};
}
std::vector<Tensor> Linear::State::storage() const {
    auto result = impl_->mixer.storage();
    const auto& p = impl_->projected;
    for (const auto* value : {&p.qk, &p.value, &p.qkv, &p.gate, &p.beta})
        if (value->defined()) result.emplace_back(*value);
    return result;
}
struct Linear::Impl {
    cuda::LinearCore core;
    Device target;
    Impl(LinearWeights w, LinearConfig c) : target(c.target) {
        std::optional<MfqCudaGuard> guard;
        if (target.is_cuda()) guard.emplace(target);
        core.key_heads=c.key_heads; core.value_heads=c.value_heads;
        core.key_dim=c.key_dim; core.value_dim=c.value_dim; core.conv_kernel=c.conv_kernel;
        core.rms_norm_eps=c.rms_norm_eps; core.norm_weight_offset=c.norm_weight_offset;
        core.split_in_proj=c.split_in_proj; core.split_dense_zab=c.split_dense_zab;
        core.dense_ab_tail=c.dense_ab_tail; core.ab_is_nint=c.ab_is_nint;
        core.dense_out_proj=c.dense_out_proj; core.tiled_v_heads=c.tiled_v_heads;
        core.attn_norm=std::move(w.attn_norm); core.conv_weight=std::move(w.conv_weight);
        core.conv_bias=std::move(w.conv_bias); core.dt_bias=std::move(w.dt_bias);
        core.a_log=std::move(w.a_log); core.linear_norm=std::move(w.linear_norm);
        core.in_proj=std::move(w.in_proj); core.qkv_proj=std::move(w.qkv_proj);
        core.ab_nint_proj=std::move(w.ab_nint_proj); core.z_proj=std::move(w.z_proj);
        if (!w.ab_proj.empty()) core.ab_proj=make_dense_group(w.ab_proj);
        if (!w.zab_proj.empty()) core.zab_proj=make_dense_group(w.zab_proj);
        core.out_proj=std::move(w.out_proj); core.out_proj_dense=std::move(w.out_proj_dense);
    }
};
Linear::Linear(LinearWeights weights, LinearConfig config)
    : impl_(std::make_unique<Impl>(std::move(weights),config)) {}
Linear::~Linear() = default;
Linear::Linear(Linear&&) noexcept = default;
Linear& Linear::operator=(Linear&&) noexcept = default;
std::size_t Linear::growth(const State& state, std::int64_t batch, std::int64_t tokens,
                           std::int64_t confirmed, bool reset) const {
    if (batch <= 0 || tokens <= 0 || confirmed < 0 || confirmed >= tokens)
        throw std::invalid_argument("invalid linear-attention growth geometry");
    if (state.pending()) throw std::runtime_error("linear attention has a pending speculative pass");
    if (reset && confirmed) throw std::invalid_argument("speculative linear attention requires existing history");
    const auto view = state.view();
    const bool retained = state.defined() && (!reset ||
        (view.convolution.size(0) == batch && view.recurrent.size(0) == batch));
    const auto mul = [](std::size_t a, std::size_t b) {
        if (b && a > std::numeric_limits<std::size_t>::max()/b) throw std::overflow_error("linear-attention growth overflow");
        return a*b;
    };
    const auto add = [](std::size_t a, std::size_t b) {
        if (a > std::numeric_limits<std::size_t>::max()-b) throw std::overflow_error("linear-attention growth overflow");
        return a+b;
    };
    const auto& c=impl_->core;
    const auto width=add(mul(2,mul(c.key_heads,c.key_dim)),mul(c.value_heads,c.value_dim));
    const auto conv=mul(mul(mul(batch,c.conv_kernel-1),width),4);
    const auto recurrent=mul(mul(mul(mul(batch,c.value_heads),c.value_dim),c.value_dim),4);
    std::size_t result=retained?0:add(conv,recurrent);
    if (!confirmed) return result;
    if (retained) {
        result=add(result,mul(view.convolution.numel(),view.convolution.element_size()));
        result=add(result,mul(view.recurrent.numel(),view.recurrent.element_size()));
    } else result=add(result,add(conv,recurrent));
    std::size_t projected=0;
    const auto count=[&](const std::vector<Tensor>& weights) {
        for (const auto& weight:weights) projected=add(projected,weight.size(0));
    };
    count(c.in_proj); count(c.qkv_proj); count(c.ab_nint_proj);
    if (c.z_proj) projected=add(projected,c.z_proj.size(0));
    for (const auto n:c.ab_proj.outs) projected=add(projected,n);
    for (const auto n:c.zab_proj.outs) projected=add(projected,n);
    // A grouped projection can retain non-tape outputs through shared backing;
    // a non-split qkv cast can additionally allocate a separate F32 tensor.
    if (!c.split_in_proj) projected=add(projected,width);
    projected=add(projected,mul(2,c.value_heads));
    return add(result,mul(mul(mul(batch,tokens),projected),4));
}
std::array<Tensor, 2> Linear::forward(State& state, const Tensor& input,
                                      std::int64_t position, std::int64_t confirmed) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->target.is_cuda()) guard.emplace(impl_->target);
    std::array<Tensor,2> result{Tensor{DenseTensor{}},Tensor{DenseTensor{}}};
    auto& executor = cuda::Executor::current();
    auto& mixer = state.impl_->mixer;
    const auto& x = input.dense();
    if (!mixer.defined()) {
        const auto& c = impl_->core;
        const auto width=2*c.key_heads*c.key_dim+c.value_heads*c.value_dim;
        const auto options=x.options().dtype(kFloat32);
        mixer.view()={cuda::zeros({x.size(0),c.conv_kernel-1,width},options),
            cuda::zeros({x.size(0),c.value_heads,c.value_dim,c.value_dim},options),
            x.is_cuda() && executor.mode().transposed};
    }
    if (confirmed == 0) {
        if (mixer.pending()) throw std::runtime_error("linear attention has a pending speculative pass");
    } else {
        if (!x.is_cuda() || mixer.pending() || confirmed <= 0 || confirmed >= x.size(1))
            throw std::runtime_error("invalid Qwen3.5 speculative linear-attention transaction");
        mixer.begin(position,confirmed,x.size(1));
    }
    auto& v=mixer.view();
    cuda::LinearView view{v.convolution.dense(),v.recurrent.dense(),v.transposed};
    try {
        auto output=impl_->core.forward(executor.projector(cuda::Executor::current_ffn()),executor.profiler(),
            executor.mode().linear,x,view,confirmed>0 ? &state.impl_->projected : nullptr);
        result[0].dense()=std::move(output[0]);
        result[1].dense()=std::move(output[1]);
        return result;
    } catch (...) {
        if (confirmed>0) state.abort();
        throw;
    }
}
void Linear::replay(State& state, std::int64_t retained) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->target.is_cuda()) guard.emplace(impl_->target);
    auto& v=state.impl_->mixer.view();
    impl_->core.replay({v.convolution.dense(),v.recurrent.dense(),v.transposed},state.impl_->projected,retained);
}
}
#endif
