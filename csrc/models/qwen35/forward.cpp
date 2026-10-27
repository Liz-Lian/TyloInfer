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

Tensor forward_layer(Collectives& collectives, const Layer& layer, LayerState& state,
                     mfq::cuda::Executor& executor, std::size_t scratch_offset, Tensor input,
                     const AttentionStep& step, std::int64_t confirmed_prefix,
                     const RopeCache& rope, const Options& options) {
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
        return executor.run(tasks, std::move(inputs), options.ffn);
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
