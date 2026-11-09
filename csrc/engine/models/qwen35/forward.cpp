#include "engine/models/qwen35/forward.h"

#include "engine/backends/cuda/native/collectives.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/ops/tensor.h"
#endif

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda::qwen35 {

namespace {

void require_forward(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

DenseTensor normalize_ids(Collectives& collectives, int primary, DenseTensor ids) {
    require_forward(ids.dim() == 1 || ids.dim() == 2,
                    "token IDs must have shape [tokens] or [batch,tokens]");
    if (ids.dim() == 1) ids = ids.unsqueeze(0);
    require_forward(ids.size(0) > 0 && ids.size(1) > 0,
                    "token IDs must not be empty");
    return collectives.copy(ids.to(::tyloi::kInt64), primary).contiguous();
}

std::optional<DenseTensor> decode_lengths(const State& state, const DenseTensor& ids, int primary) {
    if (state.cache_pos <= 0 || ids.size(1) != 1) return {};
    return ::tyloi::cuda::full(
        {ids.size(0)}, state.cache_pos + 1,
        ::tyloi::TensorOptions()
            .dtype(::tyloi::kInt64)
            .device(::tyloi::Device(::tyloi::kCUDA, primary)));
}

DenseTensor copy_to_layer(Collectives& collectives, DenseTensor value, const Layer& layer) {
    if (layer.cpu) return value.to(::tyloi::kCPU).contiguous();
    return collectives.copy(value, layer.device).contiguous();
}

AttentionStep layer_step(Collectives& collectives, const Layer& layer,
                         const AttentionStep& step) {
    AttentionStep local = step;
    local.positions = copy_to_layer(collectives, DenseTensor(step.positions), layer);
    if (step.sequence_lengths.has_value())
        local.sequence_lengths = copy_to_layer(collectives, DenseTensor(*step.sequence_lengths), layer);
    if (step.cache_positions.has_value())
        local.cache_positions = copy_to_layer(collectives, DenseTensor(*step.cache_positions), layer);
    return local;
}

void prepare_linear_state(const LinearAttention& attention, LinearState& state,
                          const DenseTensor& input, bool transposed) {
    if (state.mixer.defined()) return;
    const auto batch = input.size(0);
    const auto width = 2 * attention.key_heads * attention.key_dim +
                       attention.value_heads * attention.value_dim;
    const auto options = input.options().dtype(::tyloi::kFloat32);
    state.mixer.view() = {
        ::tyloi::cuda::zeros({batch, attention.conv_kernel - 1, width}, options),
        ::tyloi::cuda::zeros(
            {batch, attention.value_heads, attention.value_dim, attention.value_dim}, options),
        input.is_cuda() && transposed};
}

void clear_linear_speculative(LinearState& state) noexcept {
    state.projected = {};
    state.mixer.commit();
}

} // namespace

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor embed(const Model& model, Executor& executor, Collectives& collectives,
             int primary, const ::tyloi::Tensor& ids_handle, const Executor::Mode& mode) {
    auto ids = ids_handle.dense();
#else
DenseTensor embed(const Model& model, Executor& executor, Collectives& collectives,
             int primary, DenseTensor ids, const Executor::Mode& mode) {
#endif
    auto normalized = normalize_ids(collectives, primary, std::move(ids));
#ifdef TYLOINFER_NATIVE_CUDA
    auto scope = executor.enter(mfq_current_cuda_device(), mode.ffn);
    auto result = tyloi::ops::measure("model.embed", normalized.device(), [&] {
        return tyloi::ops::embedding(::tyloi::Tensor(std::move(normalized)), model.embed);
    });
    return tyloi::ops::contiguous(tyloi::ops::cast(result, tyloi::ScalarType::float16));
#else
    return ::tyloi::cuda::embed(model.embed, std::move(normalized), collectives,
                                executor.profiler(), primary)
        .to(::tyloi::kFloat16).contiguous();
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor normalize(const Model& model, Executor& executor, const ::tyloi::Tensor& hidden_handle,
                 std::int64_t batch, std::int64_t tokens) {
    auto hidden = hidden_handle.dense();
#else
DenseTensor normalize(const Model& model, Executor& executor, DenseTensor hidden,
                 std::int64_t batch, std::int64_t tokens) {
#endif
    hidden = hidden.reshape({batch, tokens, model.metadata.hidden_size});
#ifdef TYLOINFER_NATIVE_CUDA
    auto scope = executor.enter(mfq_current_cuda_device(), Ffn::Mode{});
    return tyloi::ops::measure("model.output_norm", hidden.device(), [&] {
        return tyloi::ops::rms_norm(::tyloi::Tensor(std::move(hidden)), model.output_norm,
                                    model.metadata.rms_norm_eps,
                                    model.metadata.norm_weight_offset);
    });
#else
    return executor.profiler().measure("model.output_norm", [&] {
        return qwen_rms_norm(hidden.reshape({-1, hidden.size(-1)}).to(::tyloi::kFloat32),
                             model.output_norm, model.metadata.rms_norm_eps,
                             model.metadata.norm_weight_offset)
            .reshape(hidden.sizes());
    });
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor softcap(const Model& model, Executor& executor, const ::tyloi::Tensor& logits_handle) {
#else
DenseTensor softcap(const Model& model, Executor& executor, DenseTensor logits) {
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    auto scope = executor.enter(mfq_current_cuda_device(), Ffn::Mode{});
    return tyloi::ops::softcap(logits_handle, model.metadata.final_logit_softcapping);
#else
    return ::tyloi::cuda::ops::softcap(std::move(logits), model.metadata.final_logit_softcapping);
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor logits(const Model& model, Executor& executor, const ::tyloi::Tensor& hidden_handle, bool capped,
              const Executor::Mode& mode) {
    auto hidden = hidden_handle.dense();
#else
DenseTensor logits(const Model& model, Executor& executor, DenseTensor hidden, bool capped,
              const Executor::Mode& mode) {
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    auto scope = executor.enter(mfq_current_cuda_device(), mode.ffn);
    auto result = tyloi::ops::measure("model.lm_head", hidden.device(), [&] {
        return tyloi::ops::project(::tyloi::Tensor(std::move(hidden)), model.lm_head);
    });
    return capped ? tyloi::ops::softcap(result, model.metadata.final_logit_softcapping)
                  : result;
#else
    auto project = executor.projector(mode.ffn);
    return ::tyloi::cuda::logits(model.lm_head, std::move(hidden), project,
                                 executor.profiler(),
                                 capped ? model.metadata.final_logit_softcapping : 0.0);
#endif
}

namespace {

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor forward_impl(const Model& model, State& state, Executor& executor,
#else
DenseTensor forward_impl(const Model& model, State& state, Executor& executor,
#endif
                    Collectives& collectives, const Executor::Mode& options, DenseTensor ids,
                    DenseTensor embeddings, std::optional<DenseTensor> positions,
                    std::optional<DenseTensor> lengths, tyloi::models::Input::Trace* trace,
                    bool advance, std::optional<DenseTensor> cache_positions,
#ifdef TYLOINFER_NATIVE_CUDA
                    ::tyloi::Tensor* raw_hidden,
#else
                    DenseTensor* raw_hidden,
#endif
                    std::int64_t confirmed, std::int64_t kv_length,
                    std::int64_t attention_parts) {
    ids = normalize_ids(collectives, options.primary, std::move(ids));
    const auto B = ids.size(0);
    const auto T = ids.size(1);
    require_forward(embeddings.dim() == 3 && embeddings.size(0) == B &&
                        embeddings.size(1) == T &&
                        embeddings.size(2) == model.metadata.hidden_size,
                    "inputs_embeds shape must match [batch,tokens,hidden_size]");
    require_forward(
        confirmed >= 0 &&
            (confirmed == 0 ||
             (confirmed < T && B == 1 && state.cache_pos > 0 && !positions.has_value() &&
              !cache_positions.has_value() &&
              supports_speculation(model))),
        "unsupported speculative backbone geometry");
    require_forward(state.speculative_start < 0,
                    "commit or roll back the pending speculative pass before forwarding");
    require_forward(state.cache_pos + T <= model.metadata.max_position_embeddings,
                    "Qwen3.5 history exceeds context capacity");
    if (state.cache_pos == 0) {
        state.reset(B);
    } else {
        require_forward(state.batch == B, "Qwen3.5 cache batch does not match input batch");
    }
    if (lengths.has_value())
        require_forward(lengths->dim() == 1 && lengths->size(0) == B,
                        "sequence lengths must have shape [batch]");
    auto physical = cache_positions.has_value()
                        ? collectives.copy(DenseTensor(*cache_positions).to(::tyloi::kInt64),
                                           options.primary)
                        : ::tyloi::cuda::arange(
                              state.cache_pos, state.cache_pos + T,
                              ::tyloi::TensorOptions()
                                  .device(::tyloi::Device(
                                      ::tyloi::kCUDA, options.primary))
                                  .dtype(::tyloi::kInt64));
    physical = physical.contiguous();
    require_forward((physical.dim() == 1 && physical.numel() == T) ||
                        (physical.dim() == 2 && physical.size(0) == B &&
                         physical.size(1) == T),
                    "cache_positions must have shape [tokens] or [batch,tokens]");
    auto semantic = positions.has_value()
                        ? collectives.copy(DenseTensor(*positions).to(::tyloi::kInt64),
                                           options.primary)
                        : (state.decode_position_delta == 0
                               ? physical
                               : physical + state.decode_position_delta);
    semantic = semantic.contiguous();
    const auto rank = semantic.dim();
#ifdef TYLOINFER_NATIVE_CUDA
    const bool has_mrope = model.rope.sections && model.rope.sections->numel() > 0;
#else
    const bool has_mrope = model.rope.sections.numel() > 0;
#endif
    require_forward(
        Config::valid_positions(rank, rank > 1 ? semantic.size(0) : 0,
                                semantic.size(-1), B, T,
                                has_mrope),
        "position_ids must have shape [tokens], [batch,tokens], or configured grid-MRoPE [3,tokens]");
    if (lengths.has_value()) {
        lengths = collectives.copy(DenseTensor(*lengths).to(::tyloi::kInt64), options.primary)
                       .contiguous();
    } else {
        lengths = decode_lengths(state, ids, options.primary);
    }

    std::optional<DenseTensor> physical_for_layers;
    if (has_mrope || positions.has_value() || cache_positions.has_value())
        physical_for_layers = physical;
#ifdef TYLOINFER_NATIVE_CUDA
    AttentionStep step{semantic, lengths ? ::tyloi::Tensor(*lengths) : ::tyloi::Tensor{},
                       physical_for_layers ? ::tyloi::Tensor(*physical_for_layers) : ::tyloi::Tensor{},
                       state.cache_pos, kv_length, attention_parts};
#else
    AttentionStep step{semantic, lengths ? MfqOptional<DenseTensor>(*lengths) : mfq_nullopt,
                       physical_for_layers, state.cache_pos, kv_length, attention_parts};
#endif
    auto hidden = collectives.copy(std::move(embeddings), options.primary).contiguous();
    if (trace)
        trace->push_back(collectives.copy(hidden, options.primary)
                             .to(::tyloi::kFloat32)
                             .contiguous()
                             .clone());
    if (confirmed > 0) {
        state.speculative_start = state.cache_pos;
        state.speculative_confirmed = confirmed;
    }
    try {
        for (std::size_t index = 0; index < model.layers.size(); ++index) {
            const auto& layer = model.layers[index];
            const auto* active_rope = layer.cpu ? &model.cpu_rope : &model.rope;
            if (!layer.cpu && !model.device_ropes.empty()) {
                const auto it = model.device_ropes.find(layer.device);
                if (it != model.device_ropes.end()) active_rope = &it->second;
            }
            hidden = forward_layer(collectives, layer, state.layers[index], executor,
                                   2 * index, std::move(hidden), step, confirmed,
                                   *active_rope, options);
            if (trace)
                trace->push_back(collectives.copy(hidden, options.primary)
                                     .to(::tyloi::kFloat32)
                                     .contiguous()
                                     .clone());
        }
    } catch (...) {
        state.abort_speculative();
        throw;
    }
    if (!positions.has_value() || advance) state.cache_pos += T;
    hidden = collectives.copy(std::move(hidden), options.primary).contiguous();
    auto raw = raw_hidden ? hidden : DenseTensor{};
    auto finalized = normalize(model, executor, std::move(hidden), B, T);
    if (raw_hidden) *raw_hidden = std::move(raw);
    return finalized;
}

} // namespace

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor forward(const Model& model, State& state, Executor& executor, Collectives& collectives,
               const Executor::Mode& options, tyloi::models::Input input) {
#else
DenseTensor forward(const Model& model, State& state, Executor& executor, Collectives& collectives,
               const Executor::Mode& options, tyloi::models::Input input) {
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    auto ids = normalize_ids(collectives, options.primary, input.ids.dense());
    auto embeddings = input.embeddings ? input.embeddings.dense()
        : embed(model, executor, collectives, options.primary, ids, options).dense();
    auto positions = input.positions ? std::optional<DenseTensor>(input.positions.dense()) : std::nullopt;
    auto lengths = input.lengths ? std::optional<DenseTensor>(input.lengths.dense()) : std::nullopt;
    auto cache_positions = input.cache_positions ? std::optional<DenseTensor>(input.cache_positions.dense()) : std::nullopt;
#else
    auto ids = normalize_ids(collectives, options.primary, std::move(input.ids));
    auto embeddings = input.embeddings ? *input.embeddings
        : embed(model, executor, collectives, options.primary, ids, options);
    auto positions = std::move(input.positions);
    auto lengths = std::move(input.lengths);
    auto cache_positions = std::move(input.cache_positions);
#endif
    return forward_impl(model, state, executor, collectives, options, std::move(ids),
                        std::move(embeddings), std::move(positions),
                        std::move(lengths), input.trace, input.advance,
                        std::move(cache_positions), input.raw_hidden,
                        input.confirmed, input.kv_length, input.attention_parts);
}

bool supports_speculation(const Model& model) noexcept {
    return !model.layers.empty() &&
           std::all_of(model.layers.begin(), model.layers.end(), [](const Layer& layer) {
               if (const auto* full = std::get_if<Full>(&layer.weights))
                   return !full->attention.sliding;
               return true;
           });
}

void replay(const Model& model, State& state, Executor& executor, std::int64_t accepted) {
    require_forward(state.speculative_start >= 0 && accepted >= 0,
                    "no speculative transaction");
    try {
        const auto keep = state.rewind(accepted);
        for (std::size_t index = 0; index < state.layers.size(); ++index) {
            auto* linear = std::get_if<LinearState>(&state.layers[index]);
            if (linear == nullptr) continue;
            MfqCudaGuard guard(model.layers[index].device);
            const auto* layer = std::get_if<Linear>(&model.layers[index].weights);
            require_forward(layer != nullptr, "Qwen3.5 recurrent layer weight kind mismatch");
            auto& state_view = linear->mixer.view();
            const LinearView linear_view{
                state_view.convolution, state_view.recurrent, state_view.transposed};
#ifdef TYLOINFER_NATIVE_CUDA
            auto scope = executor.enter(mfq_current_cuda_device(), Ffn::Mode{});
#endif
            layer->attention.replay(linear_view, linear->projected,
                                    keep - state.speculative_start);
        }
        state.commit_speculative();
        state.cache_pos = keep;
    } catch (...) {
        state.abort_speculative();
        throw;
    }
}

DenseTensor forward_layer(Collectives& collectives, const Layer& layer, LayerState& state,
                     tyloi::cuda::Executor& executor, std::size_t scratch_offset, DenseTensor input,
                     const AttentionStep& step, std::int64_t confirmed_prefix,
#ifdef TYLOINFER_NATIVE_CUDA
                     const tyloi::ops::RopeTable& rope,
#else
                     const RopeCache& rope,
#endif
                     const Executor::Mode& options) {
    MfqCudaGuard guard(layer.device);
    auto local_step = layer_step(collectives, layer, step);
    input = layer.cpu ? input.to(::tyloi::kCPU).contiguous()
                      : collectives.copy(input, layer.device).contiguous();
    const auto* full = std::get_if<Full>(&layer.weights);
    auto* trace = full != nullptr && options.trace_layer >= 0 &&
                                options.trace_layer == full->attention.layer
                            ? options.trace
                            : nullptr;
#ifdef TYLOINFER_NATIVE_CUDA
    auto scope = executor.enter(mfq_current_cuda_device(), options.ffn);
#endif
    if (full != nullptr) {
        auto* cache = std::get_if<tyloi::state::KvView>(&state);
        require_forward(cache != nullptr, "Qwen3.5 full layer state kind mismatch");
        const auto& attention = full->attention;
        const auto capacity = attention.sliding
                                  ? attention.attention_window
                                  : (options.kv_capacity > 0
                                         ? std::max<std::int64_t>(
                                               local_step.cache_position + input.size(1),
                                               options.kv_capacity)
                                         : attention.max_position_embeddings);
        if (!cache->defined()) {
            *cache = tyloi::state::allocate_kv(
                input.size(0), attention.kv_heads, capacity, attention.attention_head_dim,
                attention.sliding, input.device(),
                attention.official_bf16 ? ::tyloi::kBFloat16
                                        : ::tyloi::kFloat16,
                attention.kv_quantization);
        }
        cache->quantization = attention.kv_quantization;
#ifdef TYLOINFER_NATIVE_CUDA
        auto output = attention.forward(::tyloi::Tensor(std::move(input)), local_step, *cache,
                                        rope, trace);
        auto residual = output[0].dense();
        auto attention_output = output[1].dense();
#else
        auto output = attention.forward(executor.projector(options.ffn), executor.profiler(),
                                        executor.attention_options(), std::move(input), local_step,
                                        *cache, executor.attention_scratch(scratch_offset), rope,
                                        trace);
        auto residual = std::move(output[0]);
        auto attention_output = std::move(output[1]);
#endif
        return executor.full_ffn(full->ffn, full->ffn_norm, attention,
                                 std::move(residual), std::move(attention_output), options.ffn,
                                 trace);
    }
    const auto* linear = std::get_if<Linear>(&layer.weights);
    auto* linear_state = std::get_if<LinearState>(&state);
    require_forward(linear != nullptr && linear_state != nullptr,
                    "Qwen3.5 linear layer state kind mismatch");
    const auto& attention = linear->attention;
    prepare_linear_state(attention, *linear_state, input, options.transposed);
    if (confirmed_prefix == 0) {
        require_forward(!linear_state->mixer.pending(),
                        "linear attention has a pending speculative pass");
    } else {
        require_forward(input.is_cuda() && !linear_state->mixer.pending() &&
                            linear_state->mixer.defined() && confirmed_prefix > 0 &&
                            confirmed_prefix < input.size(1),
                        "invalid Qwen3.5 speculative linear-attention transaction");
        linear_state->mixer.begin(local_step.cache_position, confirmed_prefix, input.size(1));
    }
    auto& state_view = linear_state->mixer.view();
    const auto captured = confirmed_prefix > 0 ? &linear_state->projected : nullptr;
    const LinearView linear_view{
        state_view.convolution, state_view.recurrent, state_view.transposed};
    try {
        auto output = linear->attention.forward(
            executor.projector(options.ffn), executor.profiler(), options.linear,
            std::move(input), linear_view, captured);
        auto result = executor.linear_ffn(linear->ffn, linear->ffn_norm, linear->attention,
                                          std::move(output[0]), std::move(output[1]), options.ffn);
        if (confirmed_prefix > 0) {
            ++linear_state->speculative_projection_batches;
            ++linear_state->speculative_ffn_batches;
        }
        return result;
    } catch (...) {
        if (confirmed_prefix > 0) {
            linear_state->mixer.abort();
            clear_linear_speculative(*linear_state);
        }
        throw;
    }
}

} // namespace tyloi::cuda::qwen35
