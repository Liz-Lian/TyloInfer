#include "engine/backends/cuda/executor/moe_ffn_pipeline.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/native/scope.h"
#include "engine/models/session.h"
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
#include "engine/models/common/recurrent_attention.h"
#include "engine/models/common/transformer_layer.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::cuda::nemotron_h {
namespace {

using DenseTensor = ::tyloi::DenseTensor;

// The canonical graph returns the last K-1 convolution inputs for the next
// call. Runtime CUDA graphs retain this storage address across requests.
DenseTensor next_convolution_state(const DenseTensor& conv_input, int64_t kernel) {
    return conv_input.narrow(1, conv_input.size(1) - (kernel - 1), kernel - 1).contiguous();
}

} // namespace

static DenseTensor mamba_forward(const Mamba& weights, MambaState& activity, Executor& executor,
                            DenseTensor x, const Executor::Mode& mode) {
    const auto& [layer, mamba_heads, mamba_head_dim, ssm_state_size, mamba_groups, conv_kernel,
                 rms_norm_eps, time_step_limit_lo, time_step_limit_hi, mixer_norm, in_proj,
                 conv_weight_value, conv_bias_value, a_log_value, d_value, dt_bias_value, scan_norm_value, out_proj] = weights;
    auto& [mixer, speculative_xbc, speculative_dt] = activity;
    auto& profiler = executor.profiler();
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
    TYLOINFER_RUNTIME_CHECK(state.convolution.size(0) == B &&
                          state.convolution.size(1) == conv_kernel - 1 &&
                          state.convolution.size(2) == conv_width &&
                          state.recurrent.size(0) == B && state.recurrent.size(1) == heads &&
                          state.recurrent.size(2) == head_dim &&
                          state.recurrent.size(3) == state_width,
                      "mamba state geometry changed");
#ifdef TYLOINFER_NATIVE_CUDA
    std::optional<DenseTensor> empty_bias;
    const auto& bias = conv_bias_value && conv_bias_value->defined()
                          ? dynamic_cast<const DenseTensor&>(*conv_bias_value)
                          : empty_bias.emplace(::tyloi::cuda::empty({0}, f32));
#else
    const auto bias = conv_bias_value.defined()
                          ? conv_bias_value
                          : ::tyloi::cuda::empty({0}, f32);
#endif
    auto xn = profiler.measure("mamba.norm", [&] {
        return qwen_rms_norm(x.reshape({B * T, H}).to(::tyloi::kFloat32),
#ifdef TYLOINFER_NATIVE_CUDA
                             *mixer_norm, rms_norm_eps, 0.0)
#else
                             mixer_norm, rms_norm_eps, 0.0)
#endif
            .reshape({B, T, H});
    });
    auto mixed = tyloi::models::selective_state_attention<DenseTensor>(
                true,
                [&]() {
                    auto projected = profiler.measure("mamba.in_proj", [&] {
                        return executor.projector(mode.ffn).run(*in_proj, xn).to(::tyloi::kFloat32);
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
                    auto convolved = profiler.measure("mamba.conv", [&] {
#ifdef TYLOINFER_NATIVE_CUDA
                        const auto& conv_weight = dynamic_cast<const DenseTensor&>(*conv_weight_value);
#else
                        const auto& conv_weight = conv_weight_value;
#endif
                        if (T == 1)
                            return ssm_conv_silu_decode_cuda(state.convolution, xbc, conv_weight,
                                                             bias);
                        auto conv_input = ::tyloi::cuda::cat({state.convolution, xbc}, 1);
                        state.convolution.copy_(next_convolution_state(conv_input, conv_kernel));
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
                    return profiler.measure("mamba.scan", [&] {
#ifdef TYLOINFER_NATIVE_CUDA
                        const auto& dt_bias = dynamic_cast<const DenseTensor&>(*dt_bias_value);
                        const auto& a_log = dynamic_cast<const DenseTensor&>(*a_log_value);
                        const auto& d = dynamic_cast<const DenseTensor&>(*d_value);
#else
                        const auto& dt_bias = dt_bias_value;
                        const auto& a_log = a_log_value;
                        const auto& d = d_value;
#endif
                        if (T == 1) {
                            auto y = selective_state_scan_decode_cuda(
                                state.recurrent, signals[0].select(1, 0).contiguous(),
                                dt.select(1, 0).contiguous(), dt_bias, a_log,
                                signals[1].select(1, 0).contiguous(),
                                signals[2].select(1, 0).contiguous(), d, time_step_limit_lo,
                                time_step_limit_hi);
                            return std::array<DenseTensor, 2>{y.unsqueeze(1), DenseTensor()};
                        }
                        auto scanned = selective_state_scan_prefill_cuda(
                            signals[0], dt, dt_bias, a_log, signals[1], signals[2], d,
                            state.recurrent, time_step_limit_lo, time_step_limit_hi);
                        return std::array<DenseTensor, 2>{std::move(scanned[0]), std::move(scanned[1])};
                    });
                },
                [&](DenseTensor scanned, DenseTensor gate) {
                    return profiler.measure("mamba.gated_norm", [&] {
#ifdef TYLOINFER_NATIVE_CUDA
                        const auto& scan_norm = dynamic_cast<const DenseTensor&>(*scan_norm_value);
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
                    return profiler.measure(
                        "mamba.out_proj",
                        [&] { return executor.projector(mode.ffn).run(*out_proj, normalized); });
                },
                [&](const DenseTensor& final_state) {
                    if (final_state.defined())
                        state.recurrent.copy_(final_state);
                });
    xn = {};
    auto rr = x.reshape({-1, H});
    auto mm = mixed.reshape({-1, H});
    if (mm.scalar_type() != rr.scalar_type())
        mm = mm.to(rr.scalar_type()).contiguous();
    return acc_cuda(rr, mm).reshape({B, T, H});
}

static void clear_speculative(MambaState& activity) noexcept {
    activity.speculative_xbc = DenseTensor();
    activity.speculative_dt = DenseTensor();
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
    const int64_t B = speculative_xbc.size(0);
    const int64_t inner = mamba_heads * mamba_head_dim;
    const int64_t groups = mamba_groups;
    const int64_t state_width = ssm_state_size;
    const auto xbc = speculative_xbc.narrow(1, 0, retained_tokens).contiguous();
    const auto dt = speculative_dt.narrow(1, 0, retained_tokens).contiguous();
#ifdef TYLOINFER_NATIVE_CUDA
    std::optional<DenseTensor> empty_bias;
    const auto& bias = conv_bias_value && conv_bias_value->defined()
                          ? dynamic_cast<const DenseTensor&>(*conv_bias_value)
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
    auto conv_input = ::tyloi::cuda::cat({state.convolution, xbc}, 1);
    state.convolution.copy_(next_convolution_state(conv_input, conv_kernel));
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& conv_weight = dynamic_cast<const DenseTensor&>(*conv_weight_value);
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
    const auto& dt_bias = dynamic_cast<const DenseTensor&>(*dt_bias_value);
    const auto& a_log = dynamic_cast<const DenseTensor&>(*a_log_value);
    const auto& d = dynamic_cast<const DenseTensor&>(*d_value);
#else
    const auto& dt_bias = dt_bias_value;
    const auto& a_log = a_log_value;
    const auto& d = d_value;
#endif
    auto scanned = selective_state_scan_prefill_cuda(x, dt, dt_bias, a_log, b, c, d,
                                                     state.recurrent, time_step_limit_lo,
                                                     time_step_limit_hi);
    state.recurrent.copy_(scanned[1]);
}

static void rollback_speculative(const Mamba& weights, MambaState& activity, int64_t keep_position) {
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

static DenseTensor moe_forward(const Moe& weights, Executor& executor, DenseTensor x,
                          const Executor::Mode& mode) {
    const auto& [layer, rms_norm_eps, ffn_norm, ffn] = weights;
    const int64_t B = x.size(0), T = x.size(1), H = x.size(2);
    auto xn = executor.profiler().measure("moe.norm", [&] {
        return qwen_rms_norm(x.reshape({B * T, H}).to(::tyloi::kFloat32),
#ifdef TYLOINFER_NATIVE_CUDA
                             *ffn_norm, rms_norm_eps, 0.0)
#else
                             ffn_norm, rms_norm_eps, 0.0)
#endif
            .reshape({B, T, H});
    });
    auto mixed = executor.profiler().measure("moe.ffn", [&] {
#ifdef TYLOINFER_NATIVE_CUDA
        return ffn.forward(xn.reshape({B * T, H}), mode.ffn).reshape({B, T, H});
#else
        return ffn.forward(xn.reshape({B * T, H}), executor, mode.ffn).reshape({B, T, H});
#endif
    });
    xn = {};
    auto rr = x.reshape({-1, H});
    auto mm = mixed.reshape({-1, H});
    if (mm.scalar_type() != rr.scalar_type())
        mm = mm.to(rr.scalar_type()).contiguous();
    return acc_cuda(rr, mm).reshape({B, T, H});
}

static DenseTensor attention_residual(Executor& executor, DenseTensor residual, DenseTensor attention_output,
                                 int64_t batch, int64_t tokens, int64_t hidden) {
    auto rr = residual.reshape({-1, hidden});
    auto aa = attention_output.reshape({-1, hidden});
    if (executor.fp32_residual()) {
        rr = rr.to(::tyloi::kFloat32);
        aa = aa.to(::tyloi::kFloat32);
    } else if (aa.scalar_type() != rr.scalar_type()) {
        aa = aa.to(rr.scalar_type()).contiguous();
    }
    return executor.profiler().measure("attention.residual", [&] {
        return acc_cuda(rr, aa).reshape({batch, tokens, hidden});
    });
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
std::shared_ptr<tyloi::Tensor> Model::embedding(const tyloi::Tensor& ids, int primary) const {
    tyloi::models::require_model(ids.dim() == 1 || ids.dim() == 2,
                               "token IDs must have shape [tokens] or [batch,tokens]");
    auto shaped = ids.dim() == 1 ? tyloi::ops::unsqueeze(ids, 0) : nullptr;
    const auto& input = shaped ? *shaped : ids;
    tyloi::models::require_model(input.size(0) > 0 && input.size(1) > 0,
                               "token IDs must not be empty");
    auto normalized = tyloi::ops::contiguous(*tyloi::ops::copy(
        *tyloi::ops::cast(input, tyloi::ScalarType::int64),
        tyloi::Device{tyloi::DeviceType::cuda, primary}));
    return tyloi::ops::measure("model.embed", normalized->device(), [&] {
        return tyloi::ops::embedding(*normalized, *embed);
    });
}
#endif

DenseTensor embed(const Model& model, Executor& executor, Collectives& collectives,
             int primary, DenseTensor ids, const Executor::Mode& mode) {
    const Task task{task::Embed{0, 0, *model.embed}, -1, {}, "model.embed"};
    Inputs inputs;
    inputs.values = {normalize_ids(collectives, primary, std::move(ids))};
    return executor.run(std::span<const Task>{&task, 1}, std::move(inputs), mode);
}

DenseTensor normalize(const Model& model, Executor& executor, DenseTensor hidden,
                 std::int64_t batch, std::int64_t tokens) {
    hidden = hidden.reshape({batch, tokens, model.metadata.hidden_size});
    const Task task{task::Norm{0, 0,
#ifdef TYLOINFER_NATIVE_CUDA
                               *model.output_norm,
#else
                               model.output_norm,
#endif
                               model.metadata.rms_norm_eps,
                               model.metadata.norm_weight_offset},
                    -1, {}, "model.output_norm"};
    Inputs inputs;
    inputs.values = {std::move(hidden)};
    return executor.run(std::span<const Task>{&task, 1}, std::move(inputs), {});
}

DenseTensor softcap(const Model& model, Executor& executor, DenseTensor value) {
    const Task task{task::Softcap{0, 0, model.metadata.final_logit_softcapping}};
    Inputs inputs;
    inputs.values = {std::move(value)};
    return executor.run(std::span<const Task>{&task, 1}, std::move(inputs), {});
}

DenseTensor logits(const Model& model, Executor& executor, DenseTensor hidden, bool capped,
              const Executor::Mode& mode) {
    std::array<Task, 2> tasks{
        Task{task::Project{0, 0, *model.lm_head}, -1, {}, "model.lm_head"},
        Task{task::Softcap{0, 0, model.metadata.final_logit_softcapping}}};
    Inputs inputs;
    inputs.values = {std::move(hidden)};
    return executor.run(std::span<const Task>{tasks.data(), capped ? tasks.size() : 1},
                        std::move(inputs), mode);
}

bool supports_speculation(const Model& model) noexcept {
    return !model.layers.empty() &&
           std::all_of(model.layers.begin(), model.layers.end(), [](const Layer& layer) {
               const auto* full = std::get_if<FullAttention>(&layer.weights);
               return full == nullptr || !full->sliding;
           });
}

DenseTensor forward_layer(const Layer& layer, LayerState& state, Executor& executor,
                     std::size_t scratch_index, DenseTensor input, const AttentionStep& step,
                     std::int64_t confirmed, const Executor::Mode& mode) {
    MfqCudaGuard guard(layer.device);
    if (const auto* mamba = std::get_if<Mamba>(&layer.weights)) {
        auto& activity = std::get<MambaState>(state);
        if (confirmed == 0)
            return mamba_forward(*mamba, activity, executor, std::move(input), mode);
        TYLOINFER_RUNTIME_CHECK(input.is_cuda() && !activity.mixer.pending() && confirmed > 0 &&
                               confirmed < input.size(1) && activity.mixer.defined(),
                               "invalid Nemotron-H speculative mamba transaction");
        activity.mixer.begin(step.cache_position, confirmed, input.size(1));
        try {
            return mamba_forward(*mamba, activity, executor, std::move(input), mode);
        } catch (...) {
            activity.mixer.abort();
            clear_speculative(activity);
            throw;
        }
    }
    if (const auto* moe = std::get_if<Moe>(&layer.weights))
        return moe_forward(*moe, executor, std::move(input), mode);
    const auto& attention = std::get<FullAttention>(layer.weights);
    auto& cache = std::get<KVCache>(state);
    const auto B = input.size(0), T = input.size(1), H = input.size(2);
    TYLOINFER_RUNTIME_CHECK(confirmed == 0 || !attention.sliding,
                            "block does not support speculative verification");
    const auto capacity = attention.sliding ? attention.attention_window
        : (mode.kv_capacity > 0 ? std::max<std::int64_t>(step.cache_position + T, mode.kv_capacity)
                                : attention.max_position_embeddings);
    if (!cache.defined())
        cache = KVCache(B, attention.kv_heads, capacity, attention.attention_head_dim,
                        attention.sliding, input.device(),
                        attention.official_bf16 ? ::tyloi::kBFloat16 :
                                                 ::tyloi::kFloat16,
                        attention.kv_quantization);
    cache.quantization = attention.kv_quantization;
#ifdef TYLOINFER_NATIVE_CUDA
    (void)scratch_index;
    DenseTensor residual;
    const Task task{task::Attention{0, 1, 2, attention, 0, &residual}, 0};
    const auto view = cache.view();
    Inputs inputs;
    inputs.values = {std::move(input), {}, {}};
    inputs.output = 2;
    inputs.kv = std::span<const AttentionKvView>{&view, 1};
    inputs.attention = step;
    inputs.rope = attention.attention_rope;
    inputs.trace = attention.layer == mode.trace_layer ? mode.trace : nullptr;
    inputs.trace_scope = 0;
    auto output = executor.run(std::span<const Task>{&task, 1}, std::move(inputs), mode);
    return attention_residual(executor, std::move(residual), std::move(output), B, T, H);
#else
    auto mixed = attention.forward(executor.projector(mode.ffn), executor.profiler(),
        executor.attention_options(), std::move(input), step, cache.view(),
        executor.attention_scratch(scratch_index), attention.attention_rope,
        attention.layer == mode.trace_layer ? mode.trace : nullptr);
    return attention_residual(executor, std::move(mixed[0]), std::move(mixed[1]), B, T, H);
#endif
}

DenseTensor forward(const Model& model, State& state, Executor& executor, Collectives& collectives,
               const Executor::Mode& mode, Input input) {
    MfqCudaGuard guard(mode.primary);
    auto ids = normalize_ids(collectives, mode.primary, DenseTensor(std::move(input.ids)));
    auto embeddings = input.embeddings.has_value() ? DenseTensor(std::move(*input.embeddings))
        : embed(model, executor, collectives, mode.primary, ids, mode);
    const auto B = ids.size(0), T = ids.size(1);
    tyloi::models::require_model(embeddings.dim() == 3 && embeddings.size(0) == B &&
                               embeddings.size(1) == T && embeddings.size(2) == model.metadata.hidden_size,
                               "inputs_embeds shape must match [batch,tokens,hidden_size]");
    tyloi::models::require_model(state.speculative_start < 0,
                               "commit or roll back the pending speculative pass before forwarding");
    tyloi::models::require_model(input.confirmed >= 0 &&
        (input.confirmed == 0 || (input.confirmed < T && B == 1 && state.cache_pos > 0 &&
         !input.positions.has_value() && !input.cache_positions.has_value() &&
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
    auto physical = input.cache_positions.has_value()
        ? collectives.copy(DenseTensor(*input.cache_positions).to(::tyloi::kInt64), mode.primary).contiguous()
        : ::tyloi::cuda::arange(state.cache_pos, state.cache_pos + T, options);
    auto semantic = input.positions.has_value()
        ? collectives.copy(DenseTensor(*input.positions).to(::tyloi::kInt64), mode.primary).contiguous()
        : (state.decode_position_delta == 0 ? physical : physical + state.decode_position_delta);
    tyloi::models::require_model((physical.dim() == 1 && physical.numel() == T) ||
        (physical.dim() == 2 && physical.size(0) == B && physical.size(1) == T),
        "cache_positions must have shape [tokens] or [batch,tokens]");
    tyloi::models::require_model((semantic.dim() == 1 && semantic.size(0) == T) ||
        (semantic.dim() == 2 && semantic.size(0) == B && semantic.size(1) == T),
        "position_ids must have shape [tokens] or [batch,tokens]");
    if (!input.lengths.has_value() && state.cache_pos > 0 && T == 1)
        input.lengths = ::tyloi::cuda::full({B}, state.cache_pos + 1, options);
    AttentionStep step{semantic, input.lengths ? MfqOptional<DenseTensor>(DenseTensor(*input.lengths)) : mfq_nullopt,
        input.cache_positions.has_value() ? MfqOptional<DenseTensor>(physical) : mfq_nullopt,
        state.cache_pos, input.kv_length, input.attention_parts};
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
        local.positions = collectives.copy(DenseTensor(step.positions), layer.device);
        if (step.sequence_lengths.has_value())
            local.sequence_lengths = collectives.copy(DenseTensor(*step.sequence_lengths), layer.device);
        if (step.cache_positions.has_value())
            local.cache_positions = collectives.copy(DenseTensor(*step.cache_positions), layer.device);
        hidden = forward_layer(layer, state.layers[index], executor, index,
            collectives.copy(std::move(hidden), layer.device), local, input.confirmed, mode);
        trace(hidden);
    }
    if (!input.positions.has_value() || input.advance) state.cache_pos += T;
    hidden = collectives.copy(std::move(hidden), mode.primary);
    auto finalized = normalize(model, executor, std::move(hidden), B, T);
    if (input.raw_hidden) *input.raw_hidden = finalized;
    return finalized;
}

void replay(const Model& model, State& state, Executor&, std::int64_t accepted) {
    tyloi::models::require_model(state.speculative_start >= 0 && accepted >= 0,
                               "no speculative transaction");
    const auto keep = state.speculative_start + state.speculative_confirmed + accepted;
    for (std::size_t index = 0; index < state.layers.size(); ++index) {
        auto* activity = std::get_if<MambaState>(&state.layers[index]);
        if (activity == nullptr) continue;
        MfqCudaGuard guard(model.layers[index].device);
        rollback_speculative(std::get<Mamba>(model.layers[index].weights), *activity, keep);
    }
    state.cache_pos = keep;
    state.speculative_start = -1;
    state.speculative_confirmed = 0;
}

struct LayerLoader : TransformerWeightLoader {
    using Ffn = tyloi::cuda::Ffn;
};

template <class MambaWeights> void convert_mamba_weights(MambaWeights& block) {
#ifdef TYLOINFER_NATIVE_CUDA
    block.conv_weight = std::make_shared<DenseTensor>(dynamic_cast<const DenseTensor&>(*block.conv_weight).to(::tyloi::kFloat32).contiguous());
    block.a_log = std::make_shared<DenseTensor>(dynamic_cast<const DenseTensor&>(*block.a_log).to(::tyloi::kFloat32).contiguous());
    block.d = std::make_shared<DenseTensor>(dynamic_cast<const DenseTensor&>(*block.d).to(::tyloi::kFloat32).contiguous());
    block.dt_bias = std::make_shared<DenseTensor>(dynamic_cast<const DenseTensor&>(*block.dt_bias).to(::tyloi::kFloat32).contiguous());
    block.scan_norm = std::make_shared<DenseTensor>(dynamic_cast<const DenseTensor&>(*block.scan_norm).to(::tyloi::kFloat32).contiguous());
    if (block.conv_bias && block.conv_bias->defined())
        block.conv_bias = std::make_shared<DenseTensor>(dynamic_cast<const DenseTensor&>(*block.conv_bias).to(::tyloi::kFloat32).contiguous());
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

Layer load_layer(Pipelines& pipelines, const ::tyloi::weights::Loader& weights,
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
            pipelines.prepare(moe.ffn);
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

void load(Model& model, const ::tyloi::weights::Loader& weights, Pipelines& pipelines,
          LayerPlacementConfig& placement, int gpu_layers, bool expert_parallel,
          bool load_layers, bool load_input_output) {
    const auto& config = model.config;
    if (expert_parallel && config.num_routed_experts <= 0)
        throw std::runtime_error("--expert-parallel requires a model with routed experts");
    placement.prepare(config.num_hidden_layers);
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
        model.output_norm = dynamic_cast<const DenseTensor&>(*weights.tensor("model.output_norm.weight"));
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
                model.layers.push_back(load_layer(pipelines, weights.for_layer(device, false),
                                                  config, index, type));
            }
        }
    } else {
        if (load_layers) throw std::invalid_argument("streamed model setup requires deferred blocks");
#ifdef TYLOINFER_NATIVE_CUDA
        model.output_norm = weights.tensor("model.output_norm.weight");
#else
        model.output_norm = dynamic_cast<const DenseTensor&>(*weights.tensor("model.output_norm.weight"));
#endif
    }
}

State::State(const Model& model) : layers(model.layers.size()) {
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
            layers[index].emplace<KVCache>();
        } else {
            layout_.push_back({layer.device, std::monostate{}});
        }
    }
}

void State::preallocate() {
    if (!::tyloi::cuda::options::kv_vram_preallocated()) return;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        auto* cache = std::get_if<KVCache>(&layers[index]);
        if (cache == nullptr || cache->defined()) continue;
        const auto& layer = layout_[index];
        const auto& full = std::get<FullLayout>(layer.mixer);
        MfqCudaGuard guard(layer.device);
        *cache = KVCache(1, full.heads, full.capacity, full.head_dim, full.sliding,
            ::tyloi::Device(::tyloi::kCUDA, layer.device),
            full.dtype, full.quantization);
    }
}

void State::reset(std::int64_t next_batch) {
    cache_pos = decode_position_delta = 0;
    speculative_start = -1;
    speculative_confirmed = 0;
    batch = next_batch;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        MfqCudaGuard guard(layout_[index].device);
        if (auto* cache = std::get_if<KVCache>(&layers[index])) {
            if (cache->defined() && cache->batch_size() != next_batch) *cache = KVCache();
        } else if (auto* mamba = std::get_if<MambaState>(&layers[index])) {
            mamba->speculative_xbc = DenseTensor();
            mamba->speculative_dt = DenseTensor();
            mamba->mixer.reset(next_batch);
        }
    }
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

std::vector<DenseTensor> State::storage() const {
    std::vector<DenseTensor> result;
    const auto append = [&result](const DenseTensor& tensor) {
        if (tensor.defined()) result.push_back(tensor);
    };
    for (const auto& layer : layers) {
        if (const auto* cache = std::get_if<KVCache>(&layer)) {
            if (cache->is_paged()) {
                append(cache->k_chunk_ptrs);
                append(cache->v_chunk_ptrs);
                append(cache->page_table);
            } else {
                append(cache->k);
                append(cache->v);
            }
        } else if (const auto* mamba = std::get_if<MambaState>(&layer)) {
            append(mamba->mixer.view().convolution);
            append(mamba->mixer.view().recurrent);
        }
    }
    return result;
}

std::vector<DenseTensor*> State::graph_state() {
    std::vector<DenseTensor*> result;
    for (auto& layer : layers) {
        auto* mamba = std::get_if<MambaState>(&layer);
        if (mamba == nullptr) continue;
        TYLOINFER_RUNTIME_CHECK(!mamba->mixer.pending() && mamba->mixer.defined(),
                                "decode warmup requires confirmed mamba state");
        result.push_back(&mamba->mixer.view().convolution);
        result.push_back(&mamba->mixer.view().recurrent);
    }
    return result;
}

TextSessionStateKind State::text_session_state_kind() const {
    if (layers.empty() || layers.size() != layout_.size()) return TextSessionStateKind::Unsupported;
    bool all_full = true;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        const auto& mixer = layout_[index].mixer;
        if (std::holds_alternative<FullLayout>(mixer) && std::holds_alternative<KVCache>(layers[index]))
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
            snapshots.push_back(capture_kv(std::get<KVCache>(layers[index]), cache_pos, saved.bytes));
        }
        return saved;
    }
    saved.payload = std::vector<HybridBlockSessionState>{};
    auto& snapshots = std::get<std::vector<HybridBlockSessionState>>(saved.payload);
    snapshots.reserve(layers.size());
    for (std::size_t index = 0; index < layers.size(); ++index) {
        MfqCudaGuard guard(layout_[index].device);
        HybridBlockSessionState snapshot;
        if (const auto* cache = std::get_if<KVCache>(&layers[index])) {
            snapshot.kind = HybridBlockSessionStateKind::FullAttention;
            snapshot.full_attention = capture_kv(*cache, cache_pos, saved.bytes);
        } else if (const auto* mamba = std::get_if<MambaState>(&layers[index])) {
            if (mamba->mixer.pending() || !mamba->mixer.defined())
                throw std::runtime_error("Nemotron-H mamba session state is unavailable");
            snapshot.kind = HybridBlockSessionStateKind::Recurrent;
            const auto mixer = mamba->mixer.snapshot();
            snapshot.convolution = std::make_shared<DenseTensor>(mixer.convolution);
            snapshot.recurrent = std::make_shared<DenseTensor>(mixer.recurrent);
            snapshot.transposed = mixer.transposed;
            saved.bytes += session_tensor_bytes(*snapshot.convolution);
            saved.bytes += session_tensor_bytes(*snapshot.recurrent);
        } else {
            snapshot.kind = HybridBlockSessionStateKind::Stateless;
        }
        snapshots.push_back(std::move(snapshot));
    }
    return saved;
}

void State::restore_text_session_state(const TextSessionState& saved) {
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
            restore_kv(std::get<KVCache>(layers[index]), (*snapshots)[index],
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
                if (!std::holds_alternative<KVCache>(layers[index]))
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
            const auto* conv = dynamic_cast<const DenseTensor*>(snapshot.convolution.get());
            const auto* recurrent = dynamic_cast<const DenseTensor*>(snapshot.recurrent.get());
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
            if (auto* cache = std::get_if<KVCache>(&layers[index])) {
                const auto& full = std::get<FullLayout>(layout_[index].mixer);
                restore_kv(*cache, snapshot.full_attention, full.head_dim, full.dtype, full.quantization);
            } else if (auto* mamba = std::get_if<MambaState>(&layers[index])) {
                mamba->mixer.restore(Recurrent::State{
                    dynamic_cast<const DenseTensor&>(*snapshot.convolution),
                    dynamic_cast<const DenseTensor&>(*snapshot.recurrent), snapshot.transposed});
                clear_speculative(*mamba);
            }
        }
    }
    cache_pos = saved.cache_pos;
    decode_position_delta = saved.decode_position_delta;
    batch = 1;
}

} // namespace tyloi::cuda::nemotron_h
