#include "models/qwen35/forward.h"

#include "native/cuda/collectives.h"

#include <algorithm>
#include <array>
#include <span>
#include <stdexcept>
#include <utility>

namespace mfq::cuda::qwen35 {

namespace {

void require_forward(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

Tensor normalize_ids(Collectives& collectives, int primary, Tensor ids) {
    require_forward(ids.dim() == 1 || ids.dim() == 2,
                    "token IDs must have shape [tokens] or [batch,tokens]");
    if (ids.dim() == 1) ids = ids.unsqueeze(0);
    require_forward(ids.size(0) > 0 && ids.size(1) > 0,
                    "token IDs must not be empty");
    return collectives.copy(ids.to(mfq_tensor_backend::kInt64), primary).contiguous();
}

std::optional<Tensor> decode_lengths(const State& state, const Tensor& ids, int primary) {
    if (state.cache_pos <= 0 || ids.size(1) != 1) return {};
    return mfq_tensor_backend::full(
        {ids.size(0)}, state.cache_pos + 1,
        mfq_tensor_backend::TensorOptions()
            .dtype(mfq_tensor_backend::kInt64)
            .device(mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, primary)));
}

Tensor copy_to_layer(Collectives& collectives, Tensor value, const Layer& layer) {
    if (layer.cpu) return value.to(mfq_tensor_backend::kCPU).contiguous();
    return collectives.copy(value, layer.device).contiguous();
}

AttentionStep layer_step(Collectives& collectives, const Layer& layer,
                         const AttentionStep& step) {
    AttentionStep local = step;
    local.positions = copy_to_layer(collectives, step.positions, layer);
    if (step.sequence_lengths.has_value())
        local.sequence_lengths = copy_to_layer(collectives, *step.sequence_lengths, layer);
    if (step.cache_positions.has_value())
        local.cache_positions = copy_to_layer(collectives, *step.cache_positions, layer);
    if (step.attention_mask.has_value())
        local.attention_mask = copy_to_layer(collectives, *step.attention_mask, layer);
    return local;
}

void prepare_linear_state(const LinearAttention& attention, LinearState& state,
                          const Tensor& input, bool transposed) {
    if (state.mixer.defined()) return;
    const auto batch = input.size(0);
    const auto width = 2 * attention.key_heads * attention.key_dim +
                       attention.value_heads * attention.value_dim;
    const auto options = input.options().dtype(mfq_tensor_backend::kFloat32);
    state.mixer.view() = {
        mfq_tensor_backend::zeros({batch, attention.conv_kernel - 1, width}, options),
        mfq_tensor_backend::zeros(
            {batch, attention.value_heads, attention.value_dim, attention.value_dim}, options),
        input.is_cuda() && transposed};
}

void clear_linear_speculative(LinearState& state) noexcept {
    state.projected = {};
    state.mixer.commit();
}

} // namespace

Tensor embed(const Model& model, Executor& executor, Collectives& collectives,
             int primary, Tensor ids, const Executor::Mode& mode) {
    auto normalized = normalize_ids(collectives, primary, std::move(ids));
    const Task task{task::Embed{0, 0, model.embed}, -1, {}, "model.embed"};
    Inputs inputs;
    inputs.values = {std::move(normalized)};
    return executor.run(std::span<const Task>{&task, 1}, std::move(inputs), mode);
}

Tensor normalize(const Model& model, Executor& executor, Tensor hidden,
                 std::int64_t batch, std::int64_t tokens) {
    hidden = hidden.reshape({batch, tokens, model.metadata.hidden_size});
    const Task task{task::Norm{0, 0, model.output_norm, model.metadata.rms_norm_eps,
                               model.metadata.norm_weight_offset},
                    -1, {}, "model.output_norm"};
    Inputs inputs;
    inputs.values = {std::move(hidden)};
    return executor.run(std::span<const Task>{&task, 1}, std::move(inputs), {});
}

Tensor softcap(const Model& model, Executor& executor, Tensor logits) {
    const Task task{task::Softcap{0, 0, model.metadata.final_logit_softcapping}};
    Inputs inputs;
    inputs.values = {std::move(logits)};
    return executor.run(std::span<const Task>{&task, 1}, std::move(inputs), {});
}

Tensor logits(const Model& model, Executor& executor, Tensor hidden, bool capped,
              const Executor::Mode& mode) {
    std::array<Task, 2> tasks{
        Task{task::Project{0, 0, model.lm_head}, -1, {}, "model.lm_head"},
        Task{task::Softcap{0, 0, model.metadata.final_logit_softcapping}}};
    Inputs inputs;
    inputs.values = {std::move(hidden)};
    return executor.run(std::span<const Task>{tasks.data(), capped ? tasks.size() : 1},
                        std::move(inputs), mode);
}

namespace {

Tensor forward_impl(const Model& model, State& state, Executor& executor,
                    Collectives& collectives, const Executor::Mode& options, Tensor ids,
                    Tensor embeddings, std::optional<Tensor> positions,
                    std::optional<Tensor> lengths, std::vector<Tensor>* trace,
                    std::optional<Tensor> attention_mask, bool advance,
                    std::optional<Tensor> cache_positions, Tensor* raw_hidden,
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
              !cache_positions.has_value() && !attention_mask.has_value() &&
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
    if (attention_mask.has_value())
        require_forward(attention_mask->dim() == 2 && attention_mask->size(0) == B &&
                            attention_mask->size(1) >= state.cache_pos + T,
                        "attention_mask must cover [batch,cached_plus_current_tokens]");

    auto physical = cache_positions.has_value()
                        ? collectives.copy(cache_positions->to(mfq_tensor_backend::kInt64),
                                           options.primary)
                        : mfq_tensor_backend::arange(
                              state.cache_pos, state.cache_pos + T,
                              mfq_tensor_backend::TensorOptions()
                                  .device(mfq_tensor_backend::Device(
                                      mfq_tensor_backend::kCUDA, options.primary))
                                  .dtype(mfq_tensor_backend::kInt64));
    physical = physical.contiguous();
    require_forward((physical.dim() == 1 && physical.numel() == T) ||
                        (physical.dim() == 2 && physical.size(0) == B &&
                         physical.size(1) == T),
                    "cache_positions must have shape [tokens] or [batch,tokens]");
    auto semantic = positions.has_value()
                        ? collectives.copy(positions->to(mfq_tensor_backend::kInt64),
                                           options.primary)
                        : (state.decode_position_delta == 0
                               ? physical
                               : physical + state.decode_position_delta);
    semantic = semantic.contiguous();
    const auto rank = semantic.dim();
    require_forward(
        Config::valid_positions(rank, rank > 1 ? semantic.size(0) : 0,
                                semantic.size(-1), B, T,
                                model.rope.sections.numel() > 0),
        "position_ids must have shape [tokens], [batch,tokens], or configured grid-MRoPE [3,tokens]");
    if (lengths.has_value()) {
        lengths = collectives.copy(lengths->to(mfq_tensor_backend::kInt64), options.primary)
                       .contiguous();
    } else {
        lengths = decode_lengths(state, ids, options.primary);
    }

    std::optional<Tensor> physical_for_layers;
    if (model.rope.sections.numel() > 0 || positions.has_value() || cache_positions.has_value())
        physical_for_layers = physical;
    AttentionStep step{semantic, lengths, physical_for_layers, {}, state.cache_pos, kv_length,
                       attention_parts};
    auto hidden = collectives.copy(std::move(embeddings), options.primary).contiguous();
    if (trace)
        trace->push_back(collectives.copy(hidden, options.primary)
                             .to(mfq_tensor_backend::kFloat32)
                             .contiguous()
                             .clone());
    if (confirmed > 0) {
        state.speculative_start = state.cache_pos;
        state.speculative_confirmed = confirmed;
    }
    try {
        for (std::size_t index = 0; index < model.layers.size(); ++index) {
            const auto& layer = model.layers[index];
            const RopeCache* active_rope = layer.cpu ? &model.cpu_rope : &model.rope;
            if (!layer.cpu && !model.device_ropes.empty()) {
                const auto it = model.device_ropes.find(layer.device);
                if (it != model.device_ropes.end()) active_rope = &it->second;
            }
            hidden = forward_layer(collectives, layer, state.layers[index], executor,
                                   2 * index, std::move(hidden), step, confirmed,
                                   *active_rope, options);
            if (trace)
                trace->push_back(collectives.copy(hidden, options.primary)
                                     .to(mfq_tensor_backend::kFloat32)
                                     .contiguous()
                                     .clone());
        }
    } catch (...) {
        state.abort_speculative();
        throw;
    }
    if (!positions.has_value() || advance) state.cache_pos += T;
    hidden = collectives.copy(std::move(hidden), options.primary).contiguous();
    auto raw = raw_hidden ? hidden : Tensor{};
    auto finalized = normalize(model, executor, std::move(hidden), B, T);
    if (raw_hidden) *raw_hidden = std::move(raw);
    return finalized;
}

} // namespace

Tensor forward(const Model& model, State& state, Executor& executor, Collectives& collectives,
               const Executor::Mode& options, Input input) {
    auto ids = normalize_ids(collectives, options.primary, std::move(input.ids));
    auto embeddings = input.embeddings.has_value()
                          ? std::move(*input.embeddings)
                          : embed(model, executor, collectives, options.primary, ids, options);
    return forward_impl(model, state, executor, collectives, options, std::move(ids),
                        std::move(embeddings), std::move(input.positions),
                        std::move(input.lengths), input.trace,
                        std::move(input.attention_mask), input.advance,
                        std::move(input.cache_positions), input.raw_hidden,
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
            const auto replay_task = layer->replay(keep - state.speculative_start);
            const LinearStateView linear_view{
                {linear->mixer.view().convolution, linear->mixer.view().recurrent,
                 linear->mixer.view().transposed},
                &linear->projected};
            Inputs inputs;
            inputs.output = std::nullopt;
            inputs.linear = std::span<const LinearStateView>{&linear_view, 1};
            executor.run(std::span<const Task>{&replay_task, 1}, std::move(inputs), {});
        }
        state.commit_speculative();
        state.cache_pos = keep;
    } catch (...) {
        state.abort_speculative();
        throw;
    }
}

Tensor forward_layer(Collectives& collectives, const Layer& layer, LayerState& state,
                     mfq::cuda::Executor& executor, std::size_t scratch_offset, Tensor input,
                     const AttentionStep& step, std::int64_t confirmed_prefix,
                     const RopeCache& rope, const Executor::Mode& options) {
    MfqCudaGuard guard(layer.device);
    auto local_step = layer_step(collectives, layer, step);
    input = layer.cpu ? input.to(mfq_tensor_backend::kCPU).contiguous()
                      : collectives.copy(input, layer.device).contiguous();
    const auto tasks = layer.tasks();
    const auto trace = options.trace_layer >= 0 && options.trace_layer == tasks[0].scope
                           ? options.trace
                           : nullptr;
    const auto execute = [&](std::span<const AttentionKvView> kv,
                             std::span<const LinearStateView> linear,
                             LinearAttention::Options linear_options) {
        Inputs inputs{{std::move(input), {}, {}}, 0, kv, local_step, &rope, trace,
                      options.trace_layer, linear, linear_options, scratch_offset};
        return executor.run(tasks, std::move(inputs), options);
    };
    if (const auto* full = std::get_if<Full>(&layer.weights)) {
        auto* cache = std::get_if<KVCache>(&state);
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
            *cache = KVCache(
                input.size(0), attention.kv_heads, capacity, attention.attention_head_dim,
                attention.sliding, input.device(),
                attention.official_bf16 ? mfq_tensor_backend::kBFloat16
                                        : mfq_tensor_backend::kFloat16,
                attention.kv_quantization);
        }
        cache->quantization = attention.kv_quantization;
        const std::array<AttentionKvView, 1> kv{cache->view()};
        return execute(kv, std::span<const LinearStateView>{}, {});
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
    const LinearStateView linear_view{
        {state_view.convolution, state_view.recurrent, state_view.transposed}, captured};
    const std::array<LinearStateView, 1> linear_views{linear_view};
    try {
        auto result = execute(std::span<const AttentionKvView>{}, linear_views, options.linear);
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

} // namespace mfq::cuda::qwen35
