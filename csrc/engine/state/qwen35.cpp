#include "engine/executor/qwen35.h"

#include "backends/cuda/engine/state/session/state.h"
#include "engine/models/qwen35/model.h"
#include "mfq_cuda_kv_quantization.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace mfq::cuda::qwen35 {

namespace {

void require_state(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool projected(const LinearInputs& value) {
    return value.qk.defined() || value.value.defined() || value.qkv.defined() ||
           value.gate.defined() || value.beta.defined() || value.batch != 0 ||
           value.tokens != 0 || value.split;
}

bool full_layout(const Run& run) {
    if (!run.weights || run.weights->layers.empty() ||
        run.states.size() != run.weights->layers.size()) return false;
    for (std::size_t index = 0; index < run.weights->layers.size(); ++index) {
        if (!std::holds_alternative<Full>(run.weights->layers[index].weights) ||
            !std::holds_alternative<KVCache>(run.states[index])) return false;
    }
    return true;
}

bool hybrid_layout(const Run& run) {
    if (!run.weights || run.weights->layers.empty() ||
        run.states.size() != run.weights->layers.size()) return false;
    for (std::size_t index = 0; index < run.weights->layers.size(); ++index) {
        const bool full = std::holds_alternative<Full>(run.weights->layers[index].weights) &&
                          std::holds_alternative<KVCache>(run.states[index]);
        const bool linear = std::holds_alternative<Linear>(run.weights->layers[index].weights) &&
                            std::holds_alternative<LinearState>(run.states[index]);
        if (!full && !linear) return false;
    }
    return true;
}

bool on_layer_device(const Tensor& tensor, const Layer& layer) {
    return layer.cpu ? !tensor.is_cuda()
                     : tensor.is_cuda() && tensor.get_device() == layer.device;
}

void validate_linear_snapshot(const Layer& model_layer, const Linear& layer,
                              const Recurrent::State& mixer, bool transposed) {
    const auto width = 2 * layer.attention.key_heads * layer.attention.key_dim +
                       layer.attention.value_heads * layer.attention.value_dim;
    if (!(mixer.defined() && mixer.transposed == transposed &&
          mixer.convolution.scalar_type() == mfq_tensor_backend::kFloat32 &&
          mixer.recurrent.scalar_type() == mfq_tensor_backend::kFloat32 &&
          mixer.convolution.dim() == 3 && mixer.convolution.size(0) == 1 &&
          mixer.convolution.size(1) == layer.attention.conv_kernel - 1 &&
          mixer.convolution.size(2) == width && mixer.recurrent.dim() == 4 &&
          mixer.recurrent.size(0) == 1 &&
          mixer.recurrent.size(1) == layer.attention.value_heads &&
          mixer.recurrent.size(2) == layer.attention.value_dim &&
          mixer.recurrent.size(3) == layer.attention.value_dim &&
          on_layer_device(mixer.convolution, model_layer) &&
          on_layer_device(mixer.recurrent, model_layer)))
        throw CudaSessionStateError("Qwen recurrent session topology changed");
}

void validate_full_snapshot(const Layer& model_layer, const Full& layer,
                            const KvSnapshot& state) {
    validate_kv(state);
    const auto& attention = layer.attention;
    const bool quantized = attention.kv_quantization.enabled();
    const auto dtype = quantized
                           ? mfq_tensor_backend::kInt32
                           : (attention.official_bf16 ? mfq_tensor_backend::kBFloat16
                                                       : mfq_tensor_backend::kFloat16);
    if (!(state.ring == attention.sliding && state.k.scalar_type() == dtype &&
          state.v.scalar_type() == dtype && on_layer_device(state.k, model_layer) &&
          on_layer_device(state.v, model_layer) &&
          state.k.size(1) == attention.kv_heads && state.v.size(1) == attention.kv_heads))
        throw CudaSessionStateError("Qwen full-attention session topology changed");
    const auto key_width = quantized
                               ? cuda_kv_packed_width(
                                     static_cast<int>(attention.attention_head_dim),
                                     attention.kv_quantization.key_bits())
                               : attention.attention_head_dim;
    const auto value_width = quantized
                                 ? cuda_kv_packed_width(
                                       static_cast<int>(attention.attention_head_dim),
                                       attention.kv_quantization.value_bits())
                                 : attention.attention_head_dim;
    if (state.k.size(3) != key_width || state.v.size(3) != value_width)
        throw CudaSessionStateError("Qwen full-attention session quantization changed");
}

void validate_full_snapshot_length(const KvSnapshot& state, std::int64_t cache_pos) {
    if ((!state.ring && (cache_pos > state.capacity || state.k.size(2) != cache_pos)) ||
        (state.ring && state.k.size(2) != state.capacity))
        throw CudaSessionStateError("Qwen full-attention session length changed");
}

void validate_cache_configuration(const Full& layer, const KVCache& cache) {
    if (!cache.defined()) return;
    if (cache.quantization.bits != layer.attention.kv_quantization.bits ||
        cache.ring != layer.attention.sliding)
        throw CudaSessionStateError("Qwen full-attention cache configuration changed");
}

} // namespace

TextSessionStateKind Run::text_session_state_kind() const {
    if (full_layout(*this)) return TextSessionStateKind::FullAttention;
    if (hybrid_layout(*this)) return TextSessionStateKind::HybridAttention;
    return TextSessionStateKind::Unsupported;
}

bool Run::supports_text_session_state() const {
    return text_session_state_kind() != TextSessionStateKind::Unsupported;
}

bool Run::supports_paged_text_session_state() const {
    if (!full_layout(*this)) return false;
    return std::all_of(weights->layers.begin(), weights->layers.end(), [](const Layer& layer) {
        return !std::get<Full>(layer.weights).attention.sliding;
    });
}

TextSessionState Run::capture_text_session_state(
    const std::vector<std::int64_t>& tokens) const {
    const auto kind = text_session_state_kind();
    require_state(kind != TextSessionStateKind::Unsupported && batch == 1 && cache_pos > 0 &&
                      static_cast<std::size_t>(cache_pos) == tokens.size() &&
                      speculative_start < 0 && speculative_confirmed == 0,
                  "Qwen text session state is unavailable");

    TextSessionState state;
    state.tokens = tokens;
    state.cache_pos = cache_pos;
    state.decode_position_delta = decode_position_delta;
    if (kind == TextSessionStateKind::FullAttention) {
        state.payload = std::vector<KvSnapshot>{};
        auto& layers = std::get<std::vector<KvSnapshot>>(state.payload);
        layers.reserve(weights->layers.size());
        for (std::size_t index = 0; index < weights->layers.size(); ++index) {
            const auto& layer = std::get<Full>(weights->layers[index].weights);
            const auto& cache = std::get<KVCache>(states[index]);
            validate_cache_configuration(layer, cache);
            MfqCudaGuard guard(weights->layers[index].device);
            auto saved = capture_kv(cache, cache_pos, state.bytes);
            validate_full_snapshot(weights->layers[index], layer, saved);
            validate_full_snapshot_length(saved, cache_pos);
            layers.push_back(std::move(saved));
        }
        return state;
    }

    require_state(execution != nullptr, "Qwen recurrent execution context is unavailable");
    state.payload = std::vector<HybridBlockSessionState>{};
    auto& layers = std::get<std::vector<HybridBlockSessionState>>(state.payload);
    layers.reserve(weights->layers.size());
    for (std::size_t index = 0; index < weights->layers.size(); ++index) {
        const auto& model_layer = weights->layers[index];
        MfqCudaGuard guard(model_layer.device);
        HybridBlockSessionState saved;
        if (const auto* full = std::get_if<Full>(&model_layer.weights)) {
            const auto& cache = std::get<KVCache>(states[index]);
            validate_cache_configuration(*full, cache);
            saved.kind = HybridBlockSessionStateKind::FullAttention;
            saved.full_attention = capture_kv(cache, cache_pos, state.bytes);
            validate_full_snapshot(model_layer, *full, saved.full_attention);
            validate_full_snapshot_length(saved.full_attention, cache_pos);
        } else {
            const auto& linear = std::get<LinearState>(states[index]);
            require_state(!linear.mixer.pending() && !projected(linear.projected),
                          "Qwen recurrent session state has a pending speculative pass");
            const auto snapshot = linear.mixer.snapshot();
            validate_linear_snapshot(model_layer, std::get<Linear>(model_layer.weights), snapshot,
                                     !model_layer.cpu && execution->config.gdn_transposed_state);
            saved.kind = HybridBlockSessionStateKind::Recurrent;
            saved.mixer = snapshot;
            state.bytes += session_tensor_bytes(snapshot.convolution);
            state.bytes += session_tensor_bytes(snapshot.recurrent);
        }
        layers.push_back(std::move(saved));
    }
    return state;
}

void Run::restore_text_session_state(const TextSessionState& state) {
    const auto kind = text_session_state_kind();
    if (kind == TextSessionStateKind::Unsupported || state.kind() != kind || state.cache_pos <= 0 ||
        state.cache_pos > max_position_embeddings() ||
        state.tokens.size() != static_cast<std::size_t>(state.cache_pos) ||
        speculative_start >= 0 || speculative_confirmed != 0) {
        throw CudaSessionStateError("Qwen text session state is incompatible");
    }

    if (kind == TextSessionStateKind::FullAttention) {
        const auto* layers = std::get_if<std::vector<KvSnapshot>>(&state.payload);
        if (layers == nullptr || layers->size() != weights->layers.size())
            throw CudaSessionStateError("full-attention text session state is incompatible");
        for (std::size_t index = 0; index < weights->layers.size(); ++index) {
            const auto& model_layer = weights->layers[index];
            const auto& full = std::get<Full>(model_layer.weights);
            validate_cache_configuration(full, std::get<KVCache>(states[index]));
            validate_full_snapshot(model_layer, full, layers->at(index));
            validate_full_snapshot_length(layers->at(index), state.cache_pos);
        }
        for (std::size_t index = 0; index < weights->layers.size(); ++index) {
            const auto& full = std::get<Full>(weights->layers[index].weights);
            auto& cache = std::get<KVCache>(states[index]);
            MfqCudaGuard guard(weights->layers[index].device);
            restore_kv(cache, layers->at(index), full.attention.attention_head_dim,
                       full.attention.official_bf16 ? mfq_tensor_backend::kBFloat16
                                                    : mfq_tensor_backend::kFloat16,
                       full.attention.kv_quantization);
        }
    } else {
        require_state(execution != nullptr, "Qwen recurrent execution context is unavailable");
        const auto* layers = std::get_if<std::vector<HybridBlockSessionState>>(&state.payload);
        if (layers == nullptr || layers->size() != weights->layers.size())
            throw CudaSessionStateError("Qwen hybrid session state is incompatible");
        for (std::size_t index = 0; index < weights->layers.size(); ++index) {
            const auto& model_layer = weights->layers[index];
            const auto& saved = layers->at(index);
            if (const auto* full = std::get_if<Full>(&model_layer.weights)) {
                if (saved.kind != HybridBlockSessionStateKind::FullAttention)
                    throw CudaSessionStateError("Qwen hybrid full-attention layer changed");
                validate_cache_configuration(*full, std::get<KVCache>(states[index]));
                validate_full_snapshot(model_layer, *full, saved.full_attention);
                validate_full_snapshot_length(saved.full_attention, state.cache_pos);
            } else {
                if (saved.kind != HybridBlockSessionStateKind::Recurrent)
                    throw CudaSessionStateError("Qwen hybrid recurrent layer changed");
                validate_linear_snapshot(
                    model_layer, std::get<Linear>(model_layer.weights), saved.mixer,
                    !model_layer.cpu && execution->config.gdn_transposed_state);
            }
        }
        for (std::size_t index = 0; index < weights->layers.size(); ++index) {
            const auto& saved = layers->at(index);
            MfqCudaGuard guard(weights->layers[index].device);
            if (saved.kind == HybridBlockSessionStateKind::FullAttention) {
                auto& cache = std::get<KVCache>(states[index]);
                const auto& full = std::get<Full>(weights->layers[index].weights);
                restore_kv(cache, saved.full_attention, full.attention.attention_head_dim,
                           full.attention.official_bf16 ? mfq_tensor_backend::kBFloat16
                                                        : mfq_tensor_backend::kFloat16,
                           full.attention.kv_quantization);
            } else {
                auto& linear = std::get<LinearState>(states[index]);
                linear.mixer.restore(saved.mixer);
                linear.projected = {};
            }
        }
    }
    cache_pos = state.cache_pos;
    decode_position_delta = state.decode_position_delta;
    speculative_start = -1;
    speculative_confirmed = 0;
    batch = 1;
}

std::vector<Tensor*> Run::graph_state() {
    std::vector<Tensor*> result;
    for (auto& state : states) {
        auto* linear = std::get_if<LinearState>(&state);
        if (linear == nullptr) continue;
        require_state(linear->mixer.defined() && !linear->mixer.pending() &&
                          !projected(linear->projected),
                      "decode graph requires confirmed recurrent state");
        result.push_back(&linear->mixer.view().convolution);
        result.push_back(&linear->mixer.view().recurrent);
    }
    return result;
}

} // namespace mfq::cuda::qwen35
