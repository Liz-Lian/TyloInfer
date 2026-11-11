#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/native/scope.h"
#include "engine/models/session/snapshot.h"
#include "engine/ops.h"
#include "engine/backends/cuda/executor/tq_moe.h"
#include "engine/models/common/load.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/weights/loader.h"
#include "weights.h"
#include "model.h"
#include "load.h"
#include "state.h"
#include "forward.h"
#include "engine/backends/cuda/executor/options.h"

#include "engine/backends/cuda/kernels/mfq_cuda_linear_attention_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_selective_state_scan_ops.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::cuda::nemotron_h {
namespace {

using DenseTensor = ::tyloi::DenseTensor;

// Mamba-2 has the same state lifetime as GDN but different equations: the
// depthwise convolution state and the selective-scan state are separate, and
// the gated norm runs after the scan (norm_before_gate=false).
// project() returns {gate(z), convolution input(xBC), raw per-head dt};
// convolve() turns that input into {x, B, C} and advances the convolution
// state; scan() returns {y, final recurrent state} and updates the recurrent
// state in place when it is a single decode step. commit() receives the final
// state produced by a prefill scan (undefined for decode).
template <class DenseTensor, class Project, class Convolve, class Scan, class GatedNorm, class Output,
          class Commit>
DenseTensor selective_state_attention(bool cache, Project project, Convolve convolve, Scan scan,
                                GatedNorm norm_gate, Output output, Commit commit) {
    auto projected = project();
    if (projected.size() != 3)
        throw std::runtime_error("Mamba-2 projection requires z, xBC and dt");
    auto signals = convolve(projected);
    auto scanned = scan(signals, projected[2]);
    auto result = output(norm_gate(scanned[0], projected[0]));
    if (cache)
        commit(scanned[1]);
    return result;
}

// The canonical graph returns the last K-1 convolution inputs for the next
// call. Runtime CUDA graphs retain this storage address across requests.
DenseTensor next_convolution_state(const DenseTensor& conv_input, int64_t kernel) {
    return conv_input.narrow(1, conv_input.size(1) - (kernel - 1), kernel - 1).contiguous();
}

} // namespace

#ifdef TYLOINFER_NATIVE_CUDA
static DenseTensor mamba_forward(const Mamba& weights, MambaState& activity,
                            DenseTensor x) {
#else
static DenseTensor mamba_forward(const Mamba& weights, MambaState& activity, Executor& executor,
                            DenseTensor x, const Executor::Mode& mode) {
#endif
    const auto& [layer, mamba_heads, mamba_head_dim, ssm_state_size, mamba_groups, conv_kernel,
                 rms_norm_eps, time_step_limit_lo, time_step_limit_hi, mixer_norm, in_proj,
                 conv_weight_value, conv_bias_value, a_log_value, d_value, dt_bias_value, scan_norm_value, out_proj] = weights;
    auto& [mixer, speculative_xbc, speculative_dt] = activity;
    auto measure = [&](const char* name, auto&& operation) {
#ifdef TYLOINFER_NATIVE_CUDA
        return tyloi::ops::measure(name, x.device(), std::forward<decltype(operation)>(operation));
#else
        return executor.profiler().measure(name, std::forward<decltype(operation)>(operation));
#endif
    };
    const auto project = [&](const DenseTensor& input, const ::tyloi::Tensor& weight) {
#ifdef TYLOINFER_NATIVE_CUDA
        return tyloi::ops::project(::tyloi::Tensor(input), weight).dense();
#else
        return executor.projector(mode.ffn).run(weight, input);
#endif
    };
    auto& state = mixer.view();
    const int64_t B = x.size(0), T = x.size(1), H = x.size(2);
    const int64_t heads = mamba_heads;
    const int64_t head_dim = mamba_head_dim;
    const int64_t groups = mamba_groups;
    const int64_t state_width = ssm_state_size;
    const int64_t inner = heads * head_dim;
    const int64_t conv_width = inner + 2 * groups * state_width;
    const int64_t projection_width = conv_width + inner + heads;
    const auto f32 = ::tyloi::TensorOptions()
                         .device(::tyloi::kCUDA)
                         .dtype(::tyloi::kFloat32);
    if (!mixer.defined()) {
        state = {
            ::tyloi::cuda::zeros({B, conv_kernel - 1, conv_width}, f32),
            ::tyloi::cuda::zeros({B, heads, head_dim, state_width}, f32), false};
    }
    auto& convolution = state.convolution.dense();
    auto& recurrent = state.recurrent.dense();
    TYLOINFER_RUNTIME_CHECK(convolution.size(0) == B &&
                          convolution.size(1) == conv_kernel - 1 &&
                          convolution.size(2) == conv_width &&
                          recurrent.size(0) == B && recurrent.size(1) == heads &&
                          recurrent.size(2) == head_dim &&
                          recurrent.size(3) == state_width,
                      "mamba state geometry changed");
#ifdef TYLOINFER_NATIVE_CUDA
    std::optional<DenseTensor> empty_bias;
    const auto& bias = conv_bias_value && conv_bias_value.defined()
                          ? conv_bias_value.dense()
                          : empty_bias.emplace(::tyloi::cuda::empty({0}, f32));
#else
    const auto bias = conv_bias_value.defined()
                          ? conv_bias_value
                          : ::tyloi::cuda::empty({0}, f32);
#endif
    auto xn = measure("mamba.norm", [&] {
#ifdef TYLOINFER_NATIVE_CUDA
        auto normalized = tyloi::ops::rms_norm(
            ::tyloi::Tensor(x.reshape({B * T, H})), mixer_norm, rms_norm_eps, 0.0);
        return tyloi::ops::reshape(normalized, {B, T, H}).dense();
#else
        return qwen_rms_norm(x.reshape({B * T, H}).to(::tyloi::kFloat32),
                             mixer_norm, rms_norm_eps, 0.0)
            .reshape({B, T, H});
#endif
    });
    auto mixed = selective_state_attention<DenseTensor>(
                true,
                [&]() {
                    auto projected = measure("mamba.in_proj", [&] {
                        return project(xn, in_proj).to(::tyloi::kFloat32);
                    });
                    TYLOINFER_RUNTIME_CHECK(projected.size(-1) == projection_width,
                                      "mamba fused projection width disagrees with config");
                    auto gate = projected.narrow(2, 0, inner).contiguous();
                    auto xbc = projected.narrow(2, inner, conv_width).contiguous();
                    auto dt = projected.narrow(2, inner + conv_width, heads).contiguous();
                    if (mixer.pending()) {
                        speculative_xbc = xbc;
                        speculative_dt = dt;
                    }
                    return std::array<DenseTensor, 3>{std::move(gate), std::move(xbc), std::move(dt)};
                },
                [&](const std::array<DenseTensor, 3>& projected) {
                    const auto& xbc = projected[1];
                    auto convolved = measure("mamba.conv", [&] {
#ifdef TYLOINFER_NATIVE_CUDA
                        const auto& conv_weight = conv_weight_value.dense();
#else
                        const auto& conv_weight = conv_weight_value;
#endif
                        if (T == 1)
                            return ssm_conv_silu_decode_cuda(convolution, xbc, conv_weight,
                                                             bias);
                        auto conv_input = ::tyloi::cuda::cat({convolution, xbc}, 1);
                        convolution.copy_(next_convolution_state(conv_input, conv_kernel));
                        return ssm_conv_silu_cuda(conv_input, conv_weight, bias, T);
                    });
                    auto x = convolved.narrow(2, 0, inner)
                                 .reshape({B, T, heads, head_dim})
                                 .contiguous();
                    auto b = convolved.narrow(2, inner, groups * state_width)
                                 .reshape({B, T, groups, state_width})
                                 .contiguous();
                    auto c = convolved.narrow(2, inner + groups * state_width, groups * state_width)
                                 .reshape({B, T, groups, state_width})
                                 .contiguous();
                    return std::array<DenseTensor, 3>{std::move(x), std::move(b), std::move(c)};
                },
                [&](const std::array<DenseTensor, 3>& signals, const DenseTensor& dt) {
                    return measure("mamba.scan", [&] {
#ifdef TYLOINFER_NATIVE_CUDA
                        const auto& dt_bias = dt_bias_value.dense();
                        const auto& a_log = a_log_value.dense();
                        const auto& d = d_value.dense();
#else
                        const auto& dt_bias = dt_bias_value;
                        const auto& a_log = a_log_value;
                        const auto& d = d_value;
#endif
                        if (T == 1) {
                            auto y = selective_state_scan_decode_cuda(
                                recurrent, signals[0].select(1, 0).contiguous(),
                                dt.select(1, 0).contiguous(), dt_bias, a_log,
                                signals[1].select(1, 0).contiguous(),
                                signals[2].select(1, 0).contiguous(), d, time_step_limit_lo,
                                time_step_limit_hi);
                            return std::array<DenseTensor, 2>{y.unsqueeze(1), DenseTensor()};
                        }
                        auto scanned = selective_state_scan_prefill_cuda(
                            signals[0], dt, dt_bias, a_log, signals[1], signals[2], d,
                            recurrent, time_step_limit_lo, time_step_limit_hi);
                        return std::array<DenseTensor, 2>{std::move(scanned[0]), std::move(scanned[1])};
                    });
                },
                [&](DenseTensor scanned, DenseTensor gate) {
                    return measure("mamba.gated_norm", [&] {
#ifdef TYLOINFER_NATIVE_CUDA
                        const auto& scan_norm = scan_norm_value.dense();
#else
                        const auto& scan_norm = scan_norm_value;
#endif
                        return grouped_gated_rms_norm_cuda(
                                   scanned.reshape({B * T, inner}),
                                   gate.reshape({B * T, inner}), scan_norm,
                                   inner / groups, rms_norm_eps)
                            .reshape({B, T, inner});
                    });
                },
                [&](DenseTensor normalized) {
                    return measure(
                        "mamba.out_proj",
                        [&] { return project(normalized, out_proj); });
                },
                [&](const DenseTensor& final_state) {
                    if (final_state.defined())
                        recurrent.copy_(final_state);
                });
    xn = {};
    auto rr = x.reshape({-1, H});
    auto mm = mixed.reshape({-1, H});
    if (mm.scalar_type() != rr.scalar_type())
        mm = mm.to(rr.scalar_type()).contiguous();
#ifdef TYLOINFER_NATIVE_CUDA
    return tyloi::ops::add_residual(::tyloi::Tensor(rr), ::tyloi::Tensor(mm)).dense()
        .reshape({B, T, H});
#else
    return acc_cuda(rr, mm).reshape({B, T, H});
#endif
}

static void clear_speculative(MambaState& activity) noexcept {
    activity.speculative_xbc = {};
    activity.speculative_dt = {};
    activity.mixer.commit();
}

static void replay_recurrent(const Mamba& weights, MambaState& activity, int64_t retained_tokens) {
    const auto& [layer, mamba_heads, mamba_head_dim, ssm_state_size, mamba_groups, conv_kernel,
                 rms_norm_eps, time_step_limit_lo, time_step_limit_hi, mixer_norm, in_proj,
                 conv_weight_value, conv_bias_value, a_log_value, d_value, dt_bias_value, scan_norm, out_proj] = weights;
    auto& [mixer, speculative_xbc, speculative_dt] = activity;
    TYLOINFER_RUNTIME_CHECK(retained_tokens > 0 &&
                          speculative_xbc.defined() && speculative_dt.defined() &&
                          retained_tokens <= speculative_xbc.size(1) &&
                          retained_tokens <= speculative_dt.size(1),
                      "invalid mamba recurrent replay");
    auto& state = mixer.view();
    auto& convolution = state.convolution.dense();
    auto& recurrent = state.recurrent.dense();
    const int64_t B = speculative_xbc.size(0);
    const int64_t inner = mamba_heads * mamba_head_dim;
    const int64_t groups = mamba_groups;
    const int64_t state_width = ssm_state_size;
    const auto xbc = speculative_xbc.dense().narrow(1, 0, retained_tokens).contiguous();
    const auto dt = speculative_dt.dense().narrow(1, 0, retained_tokens).contiguous();
#ifdef TYLOINFER_NATIVE_CUDA
    std::optional<DenseTensor> empty_bias;
    const auto& bias = conv_bias_value && conv_bias_value.defined()
                          ? conv_bias_value.dense()
                          : empty_bias.emplace(::tyloi::cuda::empty(
                                {0}, ::tyloi::TensorOptions()
                                         .device(::tyloi::kCUDA)
                                         .dtype(::tyloi::kFloat32)));
#else
    auto bias = conv_bias_value.defined() ? conv_bias_value
                                    : ::tyloi::cuda::empty(
                                          {0}, ::tyloi::TensorOptions()
                                                   .device(::tyloi::kCUDA)
                                                   .dtype(::tyloi::kFloat32));
#endif
    auto conv_input = ::tyloi::cuda::cat({convolution, xbc}, 1);
    convolution.copy_(next_convolution_state(conv_input, conv_kernel));
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& conv_weight = conv_weight_value.dense();
#else
    const auto& conv_weight = conv_weight_value;
#endif
    auto convolved = ssm_conv_silu_cuda(conv_input, conv_weight, bias, retained_tokens);
    auto x = convolved.narrow(2, 0, inner)
                 .reshape({B, retained_tokens, mamba_heads, mamba_head_dim})
                 .contiguous();
    auto b = convolved.narrow(2, inner, groups * state_width)
                 .reshape({B, retained_tokens, groups, state_width})
                 .contiguous();
    auto c = convolved.narrow(2, inner + groups * state_width, groups * state_width)
                 .reshape({B, retained_tokens, groups, state_width})
                 .contiguous();
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& dt_bias = dt_bias_value.dense();
    const auto& a_log = a_log_value.dense();
    const auto& d = d_value.dense();
#else
    const auto& dt_bias = dt_bias_value;
    const auto& a_log = a_log_value;
    const auto& d = d_value;
#endif
    auto scanned = selective_state_scan_prefill_cuda(x, dt, dt_bias, a_log, b, c, d,
                                                     recurrent, time_step_limit_lo,
                                                     time_step_limit_hi);
    recurrent.copy_(scanned[1]);
}

static void rollback_mamba(const Mamba& weights, MambaState& activity, int64_t keep_position) {
    auto& mixer = activity.mixer;
    try {
        const int64_t retained = mixer.rewind(keep_position);
        // Only the convolution and scan states are replayed; the normalized
        // output and the projection are not recomputed.
        replay_recurrent(weights, activity, retained);
        clear_speculative(activity);
    } catch (...) {
        mixer.abort();
        clear_speculative(activity);
        throw;
    }
}

#ifdef TYLOINFER_NATIVE_CUDA
static DenseTensor moe_forward(const Moe& weights, DenseTensor x) {
#else
static DenseTensor moe_forward(const Moe& weights, Executor& executor, DenseTensor x,
                          const Executor::Mode& mode) {
#endif
    const auto& [layer, rms_norm_eps, ffn_norm, ffn] = weights;
    const int64_t B = x.size(0), T = x.size(1), H = x.size(2);
    auto measure = [&](const char* name, auto&& operation) {
#ifdef TYLOINFER_NATIVE_CUDA
        return tyloi::ops::measure(name, x.device(), std::forward<decltype(operation)>(operation));
#else
        return executor.profiler().measure(name, std::forward<decltype(operation)>(operation));
#endif
    };
    auto xn = measure("moe.norm", [&] {
#ifdef TYLOINFER_NATIVE_CUDA
        auto normalized = tyloi::ops::rms_norm(
            ::tyloi::Tensor(x.reshape({B * T, H})), ffn_norm, rms_norm_eps, 0.0);
        return tyloi::ops::reshape(normalized, {B, T, H}).dense();
#else
        return qwen_rms_norm(x.reshape({B * T, H}).to(::tyloi::kFloat32),
                             ffn_norm, rms_norm_eps, 0.0)
            .reshape({B, T, H});
#endif
    });
    auto mixed = measure("moe.ffn", [&] {
#ifdef TYLOINFER_NATIVE_CUDA
        return ffn.forward(xn.reshape({B * T, H})).reshape({B, T, H});
#else
        return ffn.forward(xn.reshape({B * T, H}), executor, mode.ffn).reshape({B, T, H});
#endif
    });
    xn = {};
    auto rr = x.reshape({-1, H});
    auto mm = mixed.reshape({-1, H});
    if (mm.scalar_type() != rr.scalar_type())
        mm = mm.to(rr.scalar_type()).contiguous();
#ifdef TYLOINFER_NATIVE_CUDA
    return tyloi::ops::add_residual(::tyloi::Tensor(rr), ::tyloi::Tensor(mm)).dense()
        .reshape({B, T, H});
#else
    return acc_cuda(rr, mm).reshape({B, T, H});
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA
static DenseTensor attention_residual(DenseTensor residual, DenseTensor attention_output,
                                 int64_t batch, int64_t tokens, int64_t hidden) {
#else
static DenseTensor attention_residual(Executor& executor, DenseTensor residual, DenseTensor attention_output,
                                 int64_t batch, int64_t tokens, int64_t hidden) {
#endif
    auto rr = residual.reshape({-1, hidden});
    auto aa = attention_output.reshape({-1, hidden});
#ifndef TYLOINFER_NATIVE_CUDA
    if (executor.fp32_residual()) {
        rr = rr.to(::tyloi::kFloat32);
        aa = aa.to(::tyloi::kFloat32);
    } else if (aa.scalar_type() != rr.scalar_type()) {
        aa = aa.to(rr.scalar_type()).contiguous();
    }
    return executor.profiler().measure("attention.residual", [&] {
        return acc_cuda(rr, aa).reshape({batch, tokens, hidden});
    });
#else
    return tyloi::ops::measure("attention.residual", residual.device(), [&] {
        return tyloi::ops::add_residual(::tyloi::Tensor(rr), ::tyloi::Tensor(aa),
            tyloi::ops::ResidualMode::configured).dense()
            .reshape({batch, tokens, hidden});
    });
#endif
}

static DenseTensor normalize_ids(Collectives& collectives, int primary, DenseTensor ids) {
    tyloi::models::require_model(ids.dim() == 1 || ids.dim() == 2,
                               "token IDs must have shape [tokens] or [batch,tokens]");
    if (ids.dim() == 1) ids = ids.unsqueeze(0);
    tyloi::models::require_model(ids.size(0) > 0 && ids.size(1) > 0,
                               "token IDs must not be empty");
    return collectives.copy(ids.to(::tyloi::kInt64), primary).contiguous();
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor Model::embedding(const tyloi::Tensor& ids, int primary) const {
    auto& executor = Executor::current();
    auto normalized = normalize_ids(executor.collectives(), primary, ids.dense());
    auto scope = executor.enter(mfq_current_cuda_device(), Executor::current_ffn());
    return ::tyloi::ops::measure("model.embed", normalized.device(), [&] {
        return ::tyloi::ops::embedding(normalized, embed);
    });
#else
DenseTensor embed(const Model& model, Executor& executor, Collectives& collectives,
             int primary, DenseTensor ids, const Executor::Mode& mode) {
    auto normalized = normalize_ids(collectives, primary, std::move(ids));
    return executor.profiler().measure("model.embed", [&] {
        return quant_embedding_lookup(model.embed, std::move(normalized));
    });
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor Model::normalize(const ::tyloi::Tensor& hidden_handle,
                 std::int64_t batch, std::int64_t tokens) const {
    const auto& model = *this;
    auto& executor = Executor::current();
    auto hidden = hidden_handle.dense();
    hidden = hidden.reshape({batch, tokens, model.metadata.hidden_size});
    auto scope = executor.enter(mfq_current_cuda_device(), FfnMode{});
    return ::tyloi::ops::measure("model.output_norm", hidden.device(), [&] {
        return ::tyloi::ops::rms_norm(hidden, model.output_norm,
                                      model.metadata.rms_norm_eps,
                                      model.metadata.norm_weight_offset);
    });
#else
DenseTensor normalize(const Model& model, Executor& executor, DenseTensor hidden,
                 std::int64_t batch, std::int64_t tokens) {
    hidden = hidden.reshape({batch, tokens, model.metadata.hidden_size});
    return executor.profiler().measure("model.output_norm", [&] {
        return qwen_rms_norm(hidden.reshape({-1, hidden.size(-1)}).to(::tyloi::kFloat32),
                             model.output_norm, model.metadata.rms_norm_eps,
                             model.metadata.norm_weight_offset)
            .reshape(hidden.sizes());
    });
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor Model::softcap(const ::tyloi::Tensor& value_handle) const {
    const auto& model = *this;
    auto& executor = Executor::current();
    auto scope = executor.enter(mfq_current_cuda_device(), FfnMode{});
    return ::tyloi::ops::softcap(value_handle, model.metadata.final_logit_softcapping);
#else
DenseTensor softcap(const Model& model, Executor&, DenseTensor value) {
    const double cap = model.metadata.final_logit_softcapping;
    return cap > 0.0 ? ::tyloi::cuda::tanh(value / cap) * cap : std::move(value);
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor Model::logits(const ::tyloi::Tensor& hidden_handle, bool capped) const {
    const auto& model = *this;
    auto& executor = Executor::current();
    auto scope = executor.enter(mfq_current_cuda_device(), Executor::current_ffn());
    auto value = ::tyloi::ops::measure("model.lm_head", hidden_handle.device(), [&] {
        return ::tyloi::ops::project(hidden_handle, model.lm_head);
    });
    return capped ? ::tyloi::ops::softcap(value, model.metadata.final_logit_softcapping) : value;
#else
DenseTensor logits(const Model& model, Executor& executor, DenseTensor hidden, bool capped,
              const Executor::Mode& mode) {
    auto value = executor.profiler().measure("model.lm_head", [&] {
        return executor.projector(mode.ffn).run(model.lm_head, std::move(hidden));
    });
    if (capped) {
        const double cap = model.metadata.final_logit_softcapping;
        if (cap > 0.0) return ::tyloi::cuda::tanh(value / cap) * cap;
    }
    return value;
#endif
}

bool supports_speculation(const Model& model) noexcept {
    return !model.layers.empty() &&
           std::all_of(model.layers.begin(), model.layers.end(), [](const Layer& layer) {
               const auto* full = std::get_if<FullAttention>(&layer.weights);
               return full == nullptr || !full->sliding;
           });
}

#ifdef TYLOINFER_NATIVE_CUDA
DenseTensor forward_layer(const Layer& layer, LayerState& state, DenseTensor input,
                     const AttentionStep& step, std::int64_t confirmed,
                     const tyloi::models::ForwardOptions& mode) {
#else
DenseTensor forward_layer(const Layer& layer, LayerState& state, Executor& executor,
                     std::size_t scratch_index, DenseTensor input, const AttentionStep& step,
                     std::int64_t confirmed, const Executor::Mode& mode) {
#endif
    MfqCudaGuard guard(layer.device);
    if (const auto* mamba = std::get_if<Mamba>(&layer.weights)) {
        auto& activity = std::get<MambaState>(state);
#ifdef TYLOINFER_NATIVE_CUDA
        const auto run = [&] {
            return mamba_forward(*mamba, activity, std::move(input));
        };
#else
        const auto run = [&] {
            return mamba_forward(*mamba, activity, executor, std::move(input), mode);
        };
#endif
        if (confirmed == 0) return run();
        TYLOINFER_RUNTIME_CHECK(input.is_cuda() && !activity.mixer.pending() && confirmed > 0 &&
                               confirmed < input.size(1) && activity.mixer.defined(),
                               "invalid Nemotron-H speculative mamba transaction");
        activity.mixer.begin(step.cache_position, confirmed, input.size(1));
        try {
            return run();
        } catch (...) {
            activity.mixer.abort();
            clear_speculative(activity);
            throw;
        }
    }
    if (const auto* moe = std::get_if<Moe>(&layer.weights))
#ifdef TYLOINFER_NATIVE_CUDA
        return moe_forward(*moe, std::move(input));
#else
        return moe_forward(*moe, executor, std::move(input), mode);
#endif
    const auto& attention = std::get<FullAttention>(layer.weights);
    auto& cache = std::get<tyloi::state::KvView>(state);
    const auto B = input.size(0), T = input.size(1), H = input.size(2);
    TYLOINFER_RUNTIME_CHECK(confirmed == 0 || !attention.sliding,
                            "block does not support speculative verification");
    const auto capacity = attention.sliding ? attention.attention_window
        : (mode.kv_capacity > 0 ? std::max<std::int64_t>(step.cache_position + T, mode.kv_capacity)
                                : attention.max_position_embeddings);
    if (!cache.defined())
        cache = tyloi::state::allocate_kv(B, attention.kv_heads, capacity,
                        attention.attention_head_dim, attention.sliding,
                        input.device(), attention.official_bf16 ? ::tyloi::kBFloat16 :
                        ::tyloi::kFloat16, attention.kv_quantization);
    cache.quantization = attention.kv_quantization;
#ifdef TYLOINFER_NATIVE_CUDA
    auto output = attention.forward(::tyloi::Tensor(std::move(input)), step,
        cache, attention.attention_rope,
        attention.layer == mode.trace_layer ? mode.trace : nullptr);
    return attention_residual(output[0].dense(), output[1].dense(), B, T, H);
#else
    auto mixed = attention.forward(executor.projector(mode.ffn), executor.profiler(),
        executor.attention_options(), std::move(input), step, cache,
        executor.attention_scratch(scratch_index), attention.attention_rope,
        attention.layer == mode.trace_layer ? mode.trace : nullptr);
    return attention_residual(executor, std::move(mixed[0]), std::move(mixed[1]), B, T, H);
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor Model::forward(tyloi::models::Input input, tyloi::models::State& state_value,
               const tyloi::models::ForwardOptions& forward_options) const {
    auto* state_ptr = dynamic_cast<State*>(&state_value);
    tyloi::models::require_model(state_ptr && state_ptr->belongs_to(*this),
                               "Nemotron-H state belongs to a different model");
    auto& state = *state_ptr;
    const auto& model = *this;
    auto& executor = Executor::current();
    auto& collectives = executor.collectives();
    auto mode = executor.forward_mode(forward_options.primary);
    mode.ffn = Executor::current_ffn();
    mode.kv_capacity = forward_options.kv_capacity;
    mode.trace = forward_options.trace;
    mode.trace_layer = forward_options.trace_layer;
#else
DenseTensor forward(const Model& model, State& state, Executor& executor, Collectives& collectives,
               const Executor::Mode& mode, tyloi::models::Input input) {
#endif
    MfqCudaGuard guard(mode.primary);
    auto ids = normalize_ids(collectives, mode.primary, std::move(input.ids).dense());
#ifdef TYLOINFER_NATIVE_CUDA
    auto embeddings = input.embeddings ? input.embeddings.dense()
        : embedding(ids, mode.primary).dense();
#else
    auto embeddings = input.embeddings ? *input.embeddings
        : embed(model, executor, collectives, mode.primary, ids, mode);
#endif
    const auto B = ids.size(0), T = ids.size(1);
    tyloi::models::require_model(embeddings.dim() == 3 && embeddings.size(0) == B &&
                               embeddings.size(1) == T && embeddings.size(2) == model.metadata.hidden_size,
                               "inputs_embeds shape must match [batch,tokens,hidden_size]");
    tyloi::models::require_model(state.speculative_start < 0,
                               "commit or roll back the pending speculative pass before forwarding");
    tyloi::models::require_model(input.confirmed >= 0 &&
        (input.confirmed == 0 || (input.confirmed < T && B == 1 && state.cache_pos > 0 &&
         !input.positions.defined() && !input.cache_positions.defined() &&
         supports_speculation(model))),
        "unsupported speculative backbone geometry");
    if (input.confirmed > 0) {
        tyloi::models::require_model(state.cache_pos + T <= model.metadata.max_position_embeddings,
                                   "speculative pass exceeds context capacity");
        state.speculative_start = state.cache_pos;
        state.speculative_confirmed = input.confirmed;
    }
    if (state.cache_pos == 0) state.reset(B);
    const auto options = ::tyloi::TensorOptions().device(::tyloi::Device(::tyloi::kCUDA, mode.primary))
                             .dtype(::tyloi::kInt64);
    auto physical = input.cache_positions.defined()
        ? collectives.copy(input.cache_positions.dense().to(::tyloi::kInt64), mode.primary).contiguous()
        : ::tyloi::cuda::arange(state.cache_pos, state.cache_pos + T, options);
    auto semantic = input.positions.defined()
        ? collectives.copy(input.positions.dense().to(::tyloi::kInt64), mode.primary).contiguous()
        : (state.decode_position_delta == 0 ? physical : physical + state.decode_position_delta);
    tyloi::models::require_model((physical.dim() == 1 && physical.numel() == T) ||
        (physical.dim() == 2 && physical.size(0) == B && physical.size(1) == T),
        "cache_positions must have shape [tokens] or [batch,tokens]");
    tyloi::models::require_model((semantic.dim() == 1 && semantic.size(0) == T) ||
        (semantic.dim() == 2 && semantic.size(0) == B && semantic.size(1) == T),
        "position_ids must have shape [tokens] or [batch,tokens]");
    if (!input.lengths.defined() && state.cache_pos > 0 && T == 1)
        input.lengths = ::tyloi::cuda::full({B}, state.cache_pos + 1, options);
#ifdef TYLOINFER_NATIVE_CUDA
    AttentionStep step{semantic, input.lengths,
        input.cache_positions.defined() ? physical : ::tyloi::DenseTensor{},
        state.cache_pos, input.kv_length, input.attention_parts};
#else
    AttentionStep step{semantic, input.lengths ? MfqOptional<DenseTensor>(input.lengths.dense()) : mfq_nullopt,
        input.cache_positions.defined() ? MfqOptional<DenseTensor>(physical) : mfq_nullopt,
        state.cache_pos, input.kv_length, input.attention_parts};
#endif
    auto hidden = collectives.copy(std::move(embeddings), mode.primary).contiguous();
    const auto trace = [&](const DenseTensor& value) {
        if (input.trace)
            input.trace->push_back(collectives.copy(value, mode.primary).to(::tyloi::kFloat32).clone());
    };
    trace(hidden);
    for (std::size_t index = 0; index < model.layers.size(); ++index) {
        const auto& layer = model.layers[index];
        MfqCudaGuard layer_guard(layer.device);
        auto local = step;
        local.positions = collectives.copy(step.positions.dense(), layer.device);
        if (step.sequence_lengths.defined())
            local.sequence_lengths = collectives.copy(step.sequence_lengths.dense(), layer.device);
        if (step.cache_positions.defined())
            local.cache_positions = collectives.copy(step.cache_positions.dense(), layer.device);
#ifdef TYLOINFER_NATIVE_CUDA
        hidden = forward_layer(layer, state.layers[index],
            collectives.copy(std::move(hidden), layer.device), local,
            input.confirmed,
            tyloi::models::ForwardOptions{mode.kv_capacity, mode.trace, mode.trace_layer, mode.primary});
#else
        hidden = forward_layer(layer, state.layers[index], executor, index,
            collectives.copy(std::move(hidden), layer.device), local, input.confirmed, mode);
#endif
        trace(hidden);
    }
    if (!input.positions.defined() || input.advance) state.cache_pos += T;
    hidden = collectives.copy(std::move(hidden), mode.primary);
#ifdef TYLOINFER_NATIVE_CUDA
    auto finalized = normalize(hidden, B, T);
#else
    auto finalized = normalize(model, executor, std::move(hidden), B, T);
#endif
    if (input.raw_hidden) *input.raw_hidden = finalized;
    return finalized;
}

void State::rollback_speculative(std::int64_t accepted) {
    tyloi::models::require_model(speculative_start >= 0 && accepted >= 0,
                               "no speculative transaction");
    const auto keep = speculative_start + speculative_confirmed + accepted;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        auto* activity = std::get_if<MambaState>(&layers[index]);
        if (activity == nullptr) continue;
        MfqCudaGuard guard(model_->layers[index].device);
        rollback_mamba(std::get<Mamba>(model_->layers[index].weights), *activity, keep);
    }
    cache_pos = keep;
    speculative_start = -1;
    speculative_confirmed = 0;
}

struct LayerLoader : TransformerWeightLoader {
    using Ffn = tyloi::cuda::Ffn;
};

template <class MambaWeights> void convert_mamba_weights(MambaWeights& block) {
#ifdef TYLOINFER_NATIVE_CUDA
    block.conv_weight = std::make_shared<DenseTensor>(block.conv_weight.dense().to(::tyloi::kFloat32).contiguous());
    block.a_log = std::make_shared<DenseTensor>(block.a_log.dense().to(::tyloi::kFloat32).contiguous());
    block.d = std::make_shared<DenseTensor>(block.d.dense().to(::tyloi::kFloat32).contiguous());
    block.dt_bias = std::make_shared<DenseTensor>(block.dt_bias.dense().to(::tyloi::kFloat32).contiguous());
    block.scan_norm = std::make_shared<DenseTensor>(block.scan_norm.dense().to(::tyloi::kFloat32).contiguous());
    if (block.conv_bias && block.conv_bias.defined())
        block.conv_bias = std::make_shared<DenseTensor>(block.conv_bias.dense().to(::tyloi::kFloat32).contiguous());
#else
    block.conv_weight = block.conv_weight.to(::tyloi::kFloat32).contiguous();
    block.a_log = block.a_log.to(::tyloi::kFloat32).contiguous();
    block.d = block.d.to(::tyloi::kFloat32).contiguous();
    block.dt_bias = block.dt_bias.to(::tyloi::kFloat32).contiguous();
    block.scan_norm = block.scan_norm.to(::tyloi::kFloat32).contiguous();
    if (block.conv_bias.defined())
        block.conv_bias = block.conv_bias.to(::tyloi::kFloat32).contiguous();
#endif
}

void validate_components(const tyloi::ModelGraph& graph) {
    if (graph.component("vision") != nullptr &&
        cuda_model_plan(graph).vision != CudaVisionAdapter::radio)
        throw std::runtime_error(
            "Nemotron-H CUDA vision requires the radio/radio_cpe.v1 vision component");
    if (graph.component("predictor") != nullptr &&
        cuda_model_plan(graph).predictor != CudaPredictorAdapter::nemotron_h)
        throw std::runtime_error(
            "Nemotron-H CUDA multi-token prediction requires the nemotron_h_mtp predictor "
            "implementation under the predictor tensor root");
}

Layer load_layer(const ::tyloi::weights::Loader& weights,
                 const Config& config, int layer, const std::string& type,
                 std::string_view tensor_root) {
    MfqCudaGuard guard(weights.placement.device);
    Layer result;
    result.device = weights.placement.device;
    LayerLoader loader{{weights}};
    switch (Config::mixer_kind(type)) {
        case tyloi::models::nemotron_h::MixerKind::linear_attention: {
            auto& mamba = result.weights.emplace<Mamba>();
            tyloi::models::nemotron_h::load_linear_attention(mamba, loader, config, layer, tensor_root);
            convert_mamba_weights(mamba);
            break;
        }
        case tyloi::models::nemotron_h::MixerKind::full_attention: {
            auto& attention = result.weights.emplace<FullAttention>();
            attention.kv_quantization = cuda_kv_quantization();
            tyloi::models::nemotron_h::load_attention(attention, loader, config, layer, tensor_root);
            break;
        }
        case tyloi::models::nemotron_h::MixerKind::moe: {
            auto& moe = result.weights.emplace<Moe>();
            tyloi::models::nemotron_h::load_moe(moe, loader, config, layer, tensor_root);
            break;
        }
        case tyloi::models::nemotron_h::MixerKind::mlp: {
            auto& moe = result.weights.emplace<Moe>();
            tyloi::models::nemotron_h::load_dense_mlp(moe, loader, config, layer, tensor_root);
            break;
        }
    }
    return result;
}

Model read(std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
           std::int64_t context) {
    validate_model_source(*source);
    Model model;
    model.source = std::move(source);
    model.graph = tyloi::models::graph(*model.source);
    model.plan = cuda_model_plan(model.graph);
    tyloi::models::require_model(model.graph.backbone == "nemotron_h",
                               "model backbone does not match the requested causal model");
    tyloi::models::nemotron_h::load_config(definition, model.graph, model.config, model.metadata);
    validate_components(model.graph);
    tyloi::models::require_model(model.config.num_hidden_layers == model.graph.topology.text_layers,
                               "model graph/config text-layer topology mismatch");
    if (context > 0) {
        tyloi::models::require_model(context <= model.metadata.max_position_embeddings,
                                   "--ctx-size exceeds max_position_embeddings");
        model.config.max_position_embeddings = context;
        model.metadata.max_position_embeddings = context;
    }
    return model;
}

void load(Model& model, const ::tyloi::weights::Loader& weights,
          LayerPlacementConfig& placement, int gpu_layers, bool expert_parallel,
          bool load_layers, bool load_input_output) {
    const auto& config = model.config;
    if (expert_parallel && config.num_routed_experts <= 0)
        throw std::runtime_error("--expert-parallel requires a model with routed experts");
    placement.prepare(config.num_hidden_layers);
#ifdef TYLOINFER_NATIVE_CUDA
    model.primary = placement.primary_device();
#endif
    const int cpu_layers = gpu_layers < 0 ? 0 :
        static_cast<int>(std::max<int64_t>(config.num_hidden_layers - gpu_layers, 0));
    if (cpu_layers > 0) {
        if (weights.placement.tensor.enabled() || expert_parallel || placement.enabled())
            throw std::runtime_error(
                "--n-gpu-layers cannot be combined with tensor/expert/layer parallelism");
        throw std::runtime_error("--n-gpu-layers currently supports dense Qwen-style blocks only");
    }
    MfqCudaGuard guard(placement.primary_device());
    if (load_input_output) {
        const bool separate = !config.tie_word_embeddings && weights.has("model.output.weight");
        model.embed = weights.embedding("model.token_embedding.weight", separate);
#ifdef TYLOINFER_NATIVE_CUDA
        model.output_norm = weights.tensor("model.output_norm.weight");
#else
        model.output_norm = weights.tensor("model.output_norm.weight").dense();
#endif
        model.lm_head = separate ? weights.weight("model.output.weight", TensorParallelAxis::Output)
            : weights.placement.tensor.enabled()
                ? weights.weight("model.token_embedding.weight", TensorParallelAxis::Output)
                : model.embed;
        if (load_layers) {
            model.layers.reserve(static_cast<std::size_t>(config.num_hidden_layers));
            for (int index = 0; index < config.num_hidden_layers; ++index) {
                const int device = placement.device_for_layer(index);
                MfqCudaGuard layer_guard(device);
                const auto& type = model.metadata.layer_types.at(static_cast<std::size_t>(index));
                std::cerr << "loading layer " << index << ' ' << type << " CUDA" << std::endl;
                model.layers.push_back(load_layer(weights.for_layer(device, false),
                                                  config, index, type));
            }
        }
    } else {
        if (load_layers) throw std::invalid_argument("streamed model setup requires deferred blocks");
#ifdef TYLOINFER_NATIVE_CUDA
        model.output_norm = weights.tensor("model.output_norm.weight");
#else
        model.output_norm = weights.tensor("model.output_norm.weight").dense();
#endif
    }
}

State::State(const Model& model) : layers(model.layers.size()), model_(&model) {
    layout_.reserve(model.layers.size());
    for (std::size_t index = 0; index < layers.size(); ++index) {
        const auto& layer = model.layers[index];
        if (const auto* mamba = std::get_if<Mamba>(&layer.weights)) {
            layout_.push_back({layer.device, MambaLayout{mamba->mamba_heads, mamba->mamba_head_dim,
                mamba->ssm_state_size, mamba->mamba_groups, mamba->conv_kernel}});
            layers[index].emplace<MambaState>();
        } else if (const auto* full = std::get_if<FullAttention>(&layer.weights)) {
            layout_.push_back({layer.device, FullLayout{full->kv_heads, full->attention_head_dim,
                full->sliding ? full->attention_window : full->max_position_embeddings,
                full->sliding, full->official_bf16 ? ::tyloi::kBFloat16 :
                ::tyloi::kFloat16, full->kv_quantization}});
            layers[index].emplace<tyloi::state::KvView>();
        } else {
            layout_.push_back({layer.device, std::monostate{}});
        }
    }
}

void State::preallocate() {
    if (!::tyloi::cuda::options::kv_vram_preallocated()) return;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        auto* cache = std::get_if<tyloi::state::KvView>(&layers[index]);
        if (cache == nullptr || cache->defined()) continue;
        const auto& layer = layout_[index];
        const auto& full = std::get<FullLayout>(layer.mixer);
        MfqCudaGuard guard(layer.device);
        *cache = tyloi::state::allocate_kv(1, full.heads, full.capacity,
            full.head_dim, full.sliding,
            ::tyloi::Device(::tyloi::kCUDA, layer.device),
            full.dtype, full.quantization);
    }
}

void State::reset(std::int64_t next_batch) {
#ifdef TYLOINFER_NATIVE_CUDA
    graph_views_.clear();
#endif
    cache_pos = decode_position_delta = 0;
    speculative_start = -1;
    speculative_confirmed = 0;
    batch = next_batch;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        MfqCudaGuard guard(layout_[index].device);
        if (auto* cache = std::get_if<tyloi::state::KvView>(&layers[index])) {
            if (cache->defined() && cache->batch != next_batch) *cache = {};
        } else if (auto* mamba = std::get_if<MambaState>(&layers[index])) {
            mamba->speculative_xbc = {};
            mamba->speculative_dt = {};
            mamba->mixer.reset(next_batch);
        }
    }
}

std::vector<std::pair<tyloi::Device, std::size_t>> State::growth(
        std::int64_t tokens, const tyloi::models::ForwardOptions& options, std::int64_t confirmed) const {
    if (batch <= 0 || tokens <= 0 || confirmed < 0 || confirmed >= tokens || cache_pos < 0)
        throw std::invalid_argument("Nemotron-H cache planning requires a positive batch and valid token window");
    if (cache_pos > std::numeric_limits<std::int64_t>::max() - tokens)
        throw std::overflow_error("Nemotron-H cache planning position overflow");
    const auto multiply = [](std::size_t a, std::size_t b) {
        if (b && a > std::numeric_limits<std::size_t>::max() / b)
            throw std::overflow_error("Nemotron-H cache planning byte count overflow");
        return a * b;
    };
    const auto sum = [](std::size_t a, std::size_t b) {
        if (a > std::numeric_limits<std::size_t>::max() - b)
            throw std::overflow_error("Nemotron-H cache planning byte count overflow");
        return a + b;
    };
    std::vector<std::pair<tyloi::Device, std::size_t>> result;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto& layout = layout_[i];
        std::size_t bytes = 0;
        if (const auto* cache = std::get_if<tyloi::state::KvView>(&layers[i])) {
            if (!cache->defined()) {
                const auto& full = std::get<FullLayout>(layout.mixer);
                const auto capacity = full.sliding ? full.capacity :
                    (options.kv_capacity > 0 ? std::max(cache_pos + tokens, options.kv_capacity) : full.capacity);
                bytes = tyloi::state::kv_bytes(batch, full.heads, capacity, full.head_dim,
                                               full.sliding, full.dtype, full.quantization);
            }
        } else if (const auto* mamba = std::get_if<MambaState>(&layers[i])) {
            const auto& m = std::get<MambaLayout>(layout.mixer);
            const auto inner = multiply(std::size_t(m.heads), std::size_t(m.head_dim));
            const auto width = sum(inner, multiply(multiply(2, std::size_t(m.groups)), std::size_t(m.state_size)));
            const auto row = multiply(multiply(std::size_t(batch), width), sizeof(float));
            const auto convolution = multiply(row, std::size_t(m.conv_kernel - 1));
            const auto recurrent = multiply(multiply(multiply(std::size_t(batch), inner),
                                                       std::size_t(m.state_size)), sizeof(float));
            if (!mamba->mixer.defined()) bytes = sum(convolution, recurrent);
            if (tokens > 1) {
                bytes = sum(bytes, multiply(row, sum(std::size_t(tokens), std::size_t(m.conv_kernel - 1))));
                bytes = sum(bytes, recurrent);
            }
            if (confirmed > 0) {
                bytes = sum(bytes, sum(convolution, recurrent));
                const auto projection_width = sum(sum(width, inner), std::size_t(m.heads));
                bytes = sum(bytes, multiply(multiply(multiply(std::size_t(batch), std::size_t(tokens)),
                                                     projection_width), sizeof(float)));
                bytes = sum(bytes, multiply(multiply(multiply(std::size_t(batch), std::size_t(tokens)),
                                                     sum(width, std::size_t(m.heads))), sizeof(float)));
                bytes = sum(bytes, multiply(row, sum(std::size_t(tokens), std::size_t(m.conv_kernel - 1))));
                bytes = sum(bytes, recurrent);
            }
        }
        if (!bytes) continue;
        const tyloi::Device device(tyloi::DeviceType::cuda, layout.device);
        auto entry = std::find_if(result.begin(), result.end(),
            [device](const auto& item) { return item.first == device; });
        if (entry == result.end()) result.emplace_back(device, bytes);
        else entry->second = sum(entry->second, bytes);
    }
    return result;
}

void State::commit_speculative() {
    tyloi::models::require_model(speculative_start >= 0, "no speculative transaction");
    for (std::size_t index = 0; index < layers.size(); ++index) {
        MfqCudaGuard guard(layout_[index].device);
        if (auto* mamba = std::get_if<MambaState>(&layers[index]))
            clear_speculative(*mamba);
    }
    speculative_start = -1;
    speculative_confirmed = 0;
}

std::vector<::tyloi::Tensor> State::storage() const {
    std::vector<::tyloi::Tensor> result;
    const auto append = [&result](const ::tyloi::Tensor& tensor) {
        if (tensor && tensor.defined()) result.push_back(tensor);
    };
    for (const auto& layer : layers) {
        if (const auto* cache = std::get_if<tyloi::state::KvView>(&layer)) {
            if (cache->paged()) {
                append(cache->key_chunk_ptrs);
                append(cache->value_chunk_ptrs);
                append(cache->page_table);
            } else {
                append(cache->physical_key);
                append(cache->physical_value);
            }
        } else if (const auto* mamba = std::get_if<MambaState>(&layer)) {
            for (const auto& tensor : mamba->mixer.storage()) append(tensor);
            append(mamba->speculative_xbc);
            append(mamba->speculative_dt);
        }
    }
    return result;
}

std::vector<::tyloi::Tensor*> State::graph_state() {
    std::vector<::tyloi::Tensor*> result;
#ifdef TYLOINFER_NATIVE_CUDA
    std::vector<::tyloi::Tensor> views;
    views.reserve(layers.size() * 2);
#endif
    for (auto& layer : layers) {
#ifdef TYLOINFER_NATIVE_CUDA
        if (const auto* cache = std::get_if<tyloi::state::KvView>(&layer)) {
            if (cache_pos <= 0 || cache->paged() || !cache->defined() ||
                cache->physical_key.dim() != 4 || cache->physical_value.dim() != 4 ||
                cache->physical_key.size(2) <= 0 ||
                cache->physical_value.size(2) != cache->physical_key.size(2))
                throw std::runtime_error("Nemotron-H graph warmup requires prefilled non-paged KV caches with matching capacities");
            const auto capacity = cache->physical_key.size(2);
            if (!cache->ring && cache_pos >= capacity)
                throw std::runtime_error("Nemotron-H graph warmup exceeds KV cache capacity");
            const auto position = cache->ring ? cache_pos % capacity : cache_pos;
            views.push_back(tyloi::ops::slice(cache->physical_key, 2, position, position + 1));
            views.push_back(tyloi::ops::slice(cache->physical_value, 2, position, position + 1));
            continue;
        }
#endif
        auto* mamba = std::get_if<MambaState>(&layer);
        if (mamba == nullptr) continue;
        TYLOINFER_RUNTIME_CHECK(!mamba->mixer.pending() && mamba->mixer.defined(),
                                "decode warmup requires confirmed mamba state");
        result.push_back(&mamba->mixer.view().convolution);
        result.push_back(&mamba->mixer.view().recurrent);
    }
#ifdef TYLOINFER_NATIVE_CUDA
    graph_views_ = std::move(views);
    result.reserve(result.size() + graph_views_.size());
    for (auto& view : graph_views_) result.push_back(&view);
#endif
    return result;
}

TextSessionStateKind State::text_session_state_kind() const {
    if (layers.empty() || layers.size() != layout_.size()) return TextSessionStateKind::Unsupported;
    bool all_full = true;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        const auto& mixer = layout_[index].mixer;
        if (std::holds_alternative<FullLayout>(mixer) && std::holds_alternative<tyloi::state::KvView>(layers[index]))
            continue;
        all_full = false;
        if ((std::holds_alternative<MambaLayout>(mixer) &&
             std::holds_alternative<MambaState>(layers[index])) ||
            (std::holds_alternative<std::monostate>(mixer) &&
             std::holds_alternative<std::monostate>(layers[index])))
            continue;
        return TextSessionStateKind::Unsupported;
    }
    return all_full ? TextSessionStateKind::FullAttention : TextSessionStateKind::HybridAttention;
}

bool State::supports_text_session_state() const {
    return text_session_state_kind() != TextSessionStateKind::Unsupported;
}

bool State::supports_paged_text_session_state() const {
    return text_session_state_kind() == TextSessionStateKind::FullAttention &&
           std::all_of(layout_.begin(), layout_.end(), [](const Layout& layer) {
               return !std::get<FullLayout>(layer.mixer).sliding;
           });
}

TextSessionState State::capture_text_session_state(const std::vector<std::int64_t>& tokens) const {
    const auto kind = text_session_state_kind();
    if (kind == TextSessionStateKind::Unsupported)
        throw std::runtime_error("text session state is unsupported by this block layout");
    if (cache_pos <= 0 || static_cast<std::size_t>(cache_pos) != tokens.size())
        throw std::runtime_error("text session token count does not match the model cache");
    TextSessionState saved;
    saved.tokens = tokens;
    saved.cache_pos = cache_pos;
    saved.decode_position_delta = decode_position_delta;
    if (kind == TextSessionStateKind::FullAttention) {
        saved.payload = std::vector<KvSnapshot>{};
        auto& snapshots = std::get<std::vector<KvSnapshot>>(saved.payload);
        snapshots.reserve(layers.size());
        for (std::size_t index = 0; index < layers.size(); ++index) {
            MfqCudaGuard guard(layout_[index].device);
            snapshots.push_back(capture_kv(std::get<tyloi::state::KvView>(layers[index]), cache_pos, saved.bytes));
        }
        return saved;
    }
    saved.payload = std::vector<HybridBlockSessionState>{};
    auto& snapshots = std::get<std::vector<HybridBlockSessionState>>(saved.payload);
    snapshots.reserve(layers.size());
    for (std::size_t index = 0; index < layers.size(); ++index) {
        MfqCudaGuard guard(layout_[index].device);
        HybridBlockSessionState snapshot;
        if (const auto* cache = std::get_if<tyloi::state::KvView>(&layers[index])) {
            snapshot.kind = HybridBlockSessionStateKind::FullAttention;
            snapshot.full_attention = capture_kv(*cache, cache_pos, saved.bytes);
        } else if (const auto* mamba = std::get_if<MambaState>(&layers[index])) {
            if (mamba->mixer.pending() || !mamba->mixer.defined())
                throw std::runtime_error("Nemotron-H mamba session state is unavailable");
            snapshot.kind = HybridBlockSessionStateKind::Recurrent;
            const auto mixer = mamba->mixer.snapshot();
            snapshot.convolution = mixer.convolution;
            snapshot.recurrent = mixer.recurrent;
            snapshot.transposed = mixer.transposed;
            saved.bytes += session_tensor_bytes(snapshot.convolution);
            saved.bytes += session_tensor_bytes(snapshot.recurrent);
        } else {
            snapshot.kind = HybridBlockSessionStateKind::Stateless;
        }
        snapshots.push_back(std::move(snapshot));
    }
    return saved;
}

std::vector<std::pair<tyloi::Device, std::size_t>> State::restore_growth(const TextSessionState& saved) const {
    const auto kind = text_session_state_kind();
    if (kind == TextSessionStateKind::Unsupported || saved.kind() != kind || saved.cache_pos <= 0 ||
        static_cast<std::size_t>(saved.cache_pos) != saved.tokens.size())
        throw CudaSessionStateError("Nemotron-H session state is incompatible");
    std::vector<std::pair<tyloi::Device, std::size_t>> result;
    const auto add = [&](const tyloi::Tensor& source, std::size_t bytes) {
        if (!bytes) return;
        const auto device = source.device();
        auto entry = std::find_if(result.begin(), result.end(),
            [device](const auto& item) { return item.first == device; });
        if (entry == result.end()) result.emplace_back(device, bytes);
        else {
            if (entry->second > std::numeric_limits<std::size_t>::max() - bytes)
                throw std::overflow_error("Nemotron-H session planning byte count overflow");
            entry->second += bytes;
        }
    };
    const auto kv = [&](const tyloi::state::KvView& cache, const KvSnapshot& snapshot) {
        validate_kv(snapshot);
        add(snapshot.k, restore_prefix_bytes(cache.physical_key, snapshot.k, 2, snapshot.capacity));
        add(snapshot.v, restore_prefix_bytes(cache.physical_value, snapshot.v, 2, snapshot.capacity));
    };
    if (kind == TextSessionStateKind::FullAttention) {
        const auto* snapshots = std::get_if<std::vector<KvSnapshot>>(&saved.payload);
        if (!snapshots || snapshots->size() != layers.size())
            throw CudaSessionStateError("full-attention text session state is incompatible");
        for (std::size_t i = 0; i < layers.size(); ++i)
            kv(std::get<tyloi::state::KvView>(layers[i]), (*snapshots)[i]);
    } else {
        const auto* snapshots = std::get_if<std::vector<HybridBlockSessionState>>(&saved.payload);
        if (!snapshots || snapshots->size() != layers.size())
            throw CudaSessionStateError("Nemotron-H session state is incompatible");
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto& snapshot = (*snapshots)[i];
            const auto& layout = layout_[i];
            if (snapshot.kind == HybridBlockSessionStateKind::FullAttention) {
                const auto* cache = std::get_if<tyloi::state::KvView>(&layers[i]);
                if (!cache) throw CudaSessionStateError("Nemotron-H full-attention layer changed");
                kv(*cache, snapshot.full_attention);
                continue;
            }
            if (snapshot.kind == HybridBlockSessionStateKind::Stateless) {
                if (!std::holds_alternative<std::monostate>(layers[i]))
                    throw CudaSessionStateError("Nemotron-H stateless layer changed");
                continue;
            }
            const auto* m = std::get_if<MambaLayout>(&layout.mixer);
            const auto* activity = std::get_if<MambaState>(&layers[i]);
            const auto width = m ? m->heads * m->head_dim + 2 * m->groups * m->state_size : 0;
            const auto* conv = snapshot.convolution.representation<DenseTensor>();
            const auto* recurrent = snapshot.recurrent.representation<DenseTensor>();
            if (snapshot.kind != HybridBlockSessionStateKind::Recurrent || !m || !activity ||
                !conv || !recurrent || !conv->defined() || !recurrent->defined() ||
                conv->scalar_type() != ::tyloi::kFloat32 || recurrent->scalar_type() != ::tyloi::kFloat32 ||
                conv->dim() != 3 || conv->size(0) != 1 || conv->size(1) != m->conv_kernel - 1 ||
                conv->size(2) != width || recurrent->dim() != 4 || recurrent->size(0) != 1 ||
                recurrent->size(1) != m->heads || recurrent->size(2) != m->head_dim ||
                recurrent->size(3) != m->state_size || !conv->is_cuda() || !recurrent->is_cuda() ||
                conv->get_device() != layout.device || recurrent->get_device() != layout.device)
                throw CudaSessionStateError("Nemotron-H mamba session topology changed");
            add(snapshot.convolution, restore_tensor_bytes(activity->mixer.view().convolution, snapshot.convolution));
            add(snapshot.recurrent, restore_tensor_bytes(activity->mixer.view().recurrent, snapshot.recurrent));
        }
    }
    return result;
}

void State::restore_text_session_state(const TextSessionState& saved) {
#ifdef TYLOINFER_NATIVE_CUDA
    graph_views_.clear();
#endif
    const auto kind = text_session_state_kind();
    if (kind != TextSessionStateKind::HybridAttention ||
        saved.kind() != TextSessionStateKind::HybridAttention) {
        const auto* snapshots = std::get_if<std::vector<KvSnapshot>>(&saved.payload);
        if (kind != TextSessionStateKind::FullAttention || saved.kind() != kind || saved.cache_pos <= 0 ||
            static_cast<std::size_t>(saved.cache_pos) != saved.tokens.size() ||
            snapshots == nullptr || snapshots->size() != layers.size())
            throw CudaSessionStateError("full-attention text session state is incompatible");
        for (const auto& snapshot : *snapshots) validate_kv(snapshot);
        for (std::size_t index = 0; index < layers.size(); ++index) {
            MfqCudaGuard guard(layout_[index].device);
            const auto& full = std::get<FullLayout>(layout_[index].mixer);
            restore_kv(std::get<tyloi::state::KvView>(layers[index]), (*snapshots)[index],
                       full.head_dim, full.dtype, full.quantization);
        }
    } else {
        const auto* snapshots = std::get_if<std::vector<HybridBlockSessionState>>(&saved.payload);
        if (kind != TextSessionStateKind::HybridAttention || saved.kind() != kind ||
            saved.cache_pos <= 0 || static_cast<std::size_t>(saved.cache_pos) != saved.tokens.size() ||
            snapshots == nullptr || snapshots->size() != layers.size())
            throw CudaSessionStateError("Nemotron-H session state is incompatible");
        for (std::size_t index = 0; index < layers.size(); ++index) {
            const auto& snapshot = (*snapshots)[index];
            const auto& layout = layout_[index];
            MfqCudaGuard guard(layout.device);
            if (snapshot.kind == HybridBlockSessionStateKind::FullAttention) {
                if (!std::holds_alternative<tyloi::state::KvView>(layers[index]))
                    throw CudaSessionStateError("Nemotron-H full-attention layer changed");
                validate_kv(snapshot.full_attention);
                continue;
            }
            if (snapshot.kind == HybridBlockSessionStateKind::Stateless) {
                if (!std::holds_alternative<std::monostate>(layers[index]))
                    throw CudaSessionStateError("Nemotron-H stateless layer changed");
                continue;
            }
            const auto* mamba = std::get_if<MambaLayout>(&layout.mixer);
            const auto width = mamba != nullptr ? mamba->heads * mamba->head_dim +
                2 * mamba->groups * mamba->state_size : 0;
            const auto* conv = snapshot.convolution.representation<DenseTensor>();
            const auto* recurrent = snapshot.recurrent.representation<DenseTensor>();
            if (snapshot.kind != HybridBlockSessionStateKind::Recurrent || mamba == nullptr ||
                !std::holds_alternative<MambaState>(layers[index]) ||
                !conv || !recurrent || !conv->defined() || !recurrent->defined() ||
                conv->scalar_type() != ::tyloi::kFloat32 ||
                recurrent->scalar_type() != ::tyloi::kFloat32 ||
                conv->dim() != 3 || conv->size(0) != 1 ||
                conv->size(1) != mamba->conv_kernel - 1 || conv->size(2) != width ||
                recurrent->dim() != 4 || recurrent->size(0) != 1 ||
                recurrent->size(1) != mamba->heads || recurrent->size(2) != mamba->head_dim ||
                recurrent->size(3) != mamba->state_size || !conv->is_cuda() || !recurrent->is_cuda() ||
                conv->get_device() != layout.device || recurrent->get_device() != layout.device)
                throw CudaSessionStateError("Nemotron-H mamba session topology changed");
        }
        for (std::size_t index = 0; index < layers.size(); ++index) {
            const auto& snapshot = (*snapshots)[index];
            MfqCudaGuard guard(layout_[index].device);
            if (auto* cache = std::get_if<tyloi::state::KvView>(&layers[index])) {
                const auto& full = std::get<FullLayout>(layout_[index].mixer);
                restore_kv(*cache, snapshot.full_attention, full.head_dim, full.dtype, full.quantization);
            } else if (auto* mamba = std::get_if<MambaState>(&layers[index])) {
                mamba->mixer.restore(Recurrent::State{
                    snapshot.convolution,
                    snapshot.recurrent, snapshot.transposed});
                clear_speculative(*mamba);
            }
        }
    }
    cache_pos = saved.cache_pos;
    decode_position_delta = saved.decode_position_delta;
    batch = 1;
}

} // namespace tyloi::cuda::nemotron_h
