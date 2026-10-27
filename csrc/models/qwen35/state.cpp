#include "models/qwen35/state.h"

#include "models/qwen35/model.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace mfq::cuda::qwen35 {

namespace {

using Tensor = mfq_tensor_backend::Tensor;

void require_state(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool projected(const LinearInputs& value) {
    return value.qk.defined() || value.value.defined() || value.qkv.defined() ||
           value.gate.defined() || value.beta.defined() || value.batch != 0 ||
           value.tokens != 0 || value.split;
}

bool on_layer_device(const Tensor& tensor, const Layout& layer) {
    return layer.cpu ? !tensor.is_cuda()
                     : tensor.is_cuda() && tensor.get_device() == layer.device;
}

bool full_layout(const State& state) {
    if (state.layers.empty() || state.layers.size() != state.layout().size()) return false;
    for (std::size_t index = 0; index < state.layers.size(); ++index) {
        if (!std::holds_alternative<FullLayout>(state.layout()[index].attention) ||
            !std::holds_alternative<KVCache>(state.layers[index]))
            return false;
    }
    return true;
}

bool hybrid_layout(const State& state) {
    if (state.layers.empty() || state.layers.size() != state.layout().size()) return false;
    for (std::size_t index = 0; index < state.layers.size(); ++index) {
        const bool full = std::holds_alternative<FullLayout>(state.layout()[index].attention) &&
                          std::holds_alternative<KVCache>(state.layers[index]);
        const bool linear = std::holds_alternative<LinearLayout>(state.layout()[index].attention) &&
                            std::holds_alternative<LinearState>(state.layers[index]);
        if (!full && !linear) return false;
    }
    return true;
}

void validate_linear_snapshot(const Layout& model_layer, const LinearLayout& layer,
                             const Recurrent::State& mixer, bool transposed) {
    const auto width = 2 * layer.key_heads * layer.key_dim +
                       layer.value_heads * layer.value_dim;
    if (!(mixer.defined() && mixer.transposed == transposed &&
          mixer.convolution.scalar_type() == mfq_tensor_backend::kFloat32 &&
          mixer.recurrent.scalar_type() == mfq_tensor_backend::kFloat32 &&
          mixer.convolution.dim() == 3 && mixer.convolution.size(0) == 1 &&
          mixer.convolution.size(1) == layer.conv_kernel - 1 &&
          mixer.convolution.size(2) == width && mixer.recurrent.dim() == 4 &&
          mixer.recurrent.size(0) == 1 &&
          mixer.recurrent.size(1) == layer.value_heads &&
          mixer.recurrent.size(2) == layer.value_dim &&
          mixer.recurrent.size(3) == layer.value_dim &&
          on_layer_device(mixer.convolution, model_layer) &&
          on_layer_device(mixer.recurrent, model_layer)))
        throw CudaSessionStateError("Qwen recurrent session topology changed");
}

void validate_full_snapshot(const Layout& model_layer, const FullLayout& layer,
                            const KvSnapshot& state) {
    validate_kv(state);
    const bool quantized = layer.quantization.enabled();
    const auto dtype = quantized ? mfq_tensor_backend::kInt32 : layer.dtype;
    if (!(state.ring == layer.sliding && state.k.scalar_type() == dtype &&
          state.v.scalar_type() == dtype && on_layer_device(state.k, model_layer) &&
          on_layer_device(state.v, model_layer) && state.k.size(1) == layer.kv_heads &&
          state.v.size(1) == layer.kv_heads))
        throw CudaSessionStateError("Qwen full-attention session topology changed");
    const auto key_width = quantized
                               ? cuda_kv_packed_width(static_cast<int>(layer.head_dim),
                                                       layer.quantization.key_bits())
                               : layer.head_dim;
    const auto value_width = quantized
                                 ? cuda_kv_packed_width(static_cast<int>(layer.head_dim),
                                                         layer.quantization.value_bits())
                                 : layer.head_dim;
    if (state.k.size(3) != key_width || state.v.size(3) != value_width)
        throw CudaSessionStateError("Qwen full-attention session quantization changed");
}

void validate_full_snapshot_length(const KvSnapshot& state, std::int64_t cache_pos) {
    if ((!state.ring && (cache_pos > state.capacity || state.k.size(2) != cache_pos)) ||
        (state.ring && state.k.size(2) != state.capacity))
        throw CudaSessionStateError("Qwen full-attention session length changed");
}

void validate_cache_configuration(const FullLayout& layer, const KVCache& cache) {
    if (!cache.defined()) return;
    if (cache.quantization.bits != layer.quantization.bits || cache.ring != layer.sliding)
        throw CudaSessionStateError("Qwen full-attention cache configuration changed");
}

void initialize_layers(std::vector<LayerState>& layers,
                       const std::vector<Layout>& layout) {
    layers = std::vector<LayerState>(layout.size());
    for (std::size_t index = 0; index < layout.size(); ++index)
        if (std::holds_alternative<LinearLayout>(layout[index].attention))
            layers[index].emplace<LinearState>();
}

} // namespace

State::State(const Model& model, const bool& transposed)
    : max_position_embeddings_(model.metadata.max_position_embeddings),
      transposed_(&transposed) {
    layout_.reserve(model.layers.size());
    for (const auto& layer : model.layers) {
        if (const auto* full = std::get_if<Full>(&layer.weights)) {
            layout_.push_back({layer.device,
                               layer.cpu,
                               FullLayout{full->attention.kv_heads,
                                          full->attention.attention_head_dim,
                                          full->attention.sliding,
                                          full->attention.official_bf16
                                              ? mfq_tensor_backend::kBFloat16
                                              : mfq_tensor_backend::kFloat16,
                                          full->attention.kv_quantization}});
        } else {
            const auto* linear = std::get_if<Linear>(&layer.weights);
            require_state(linear != nullptr, "Qwen3.5 layer weight kind mismatch");
            layout_.push_back({layer.device,
                               layer.cpu,
                               LinearLayout{linear->attention.key_heads,
                                            linear->attention.key_dim,
                                            linear->attention.value_heads,
                                            linear->attention.value_dim,
                                            linear->attention.conv_kernel}});
        }
    }
    initialize_layers(layers, layout_);
}

void State::reset(std::int64_t new_batch) {
    require_state(new_batch > 0, "Qwen3.5 batch must be positive");
    cache_pos = 0;
    decode_position_delta = 0;
    speculative_start = -1;
    speculative_confirmed = 0;
    batch = new_batch;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        MfqCudaGuard guard(layout_[index].device);
        auto& state = layers[index];
        if (auto* cache = std::get_if<KVCache>(&state)) {
            if (cache->defined() && cache->batch_size() != new_batch) *cache = KVCache();
            continue;
        }
        auto& linear = std::get<LinearState>(state);
        linear.mixer.reset(new_batch);
        linear.projected = {};
    }
}

void State::commit_speculative() {
    require_state(speculative_start >= 0, "no speculative transaction");
    for (std::size_t index = 0; index < layers.size(); ++index) {
        MfqCudaGuard guard(layout_[index].device);
        if (auto* linear = std::get_if<LinearState>(&layers[index])) {
            linear->projected = {};
            linear->mixer.commit();
        }
    }
    speculative_start = -1;
    speculative_confirmed = 0;
}

std::int64_t State::rewind(std::int64_t accepted_suffix) {
    require_state(speculative_start >= 0 && accepted_suffix >= 0,
                  "no speculative transaction");
    const auto keep = speculative_start + speculative_confirmed + accepted_suffix;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        auto* linear = std::get_if<LinearState>(&layers[index]);
        if (linear == nullptr) continue;
        MfqCudaGuard guard(layout_[index].device);
        linear->mixer.rewind(keep);
    }
    return keep;
}

void State::abort_speculative() noexcept {
    const auto start = speculative_start;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        auto* linear = std::get_if<LinearState>(&layers[index]);
        if (linear == nullptr) continue;
        try {
            MfqCudaGuard guard(layout_[index].device);
            linear->mixer.abort();
        } catch (...) {
        }
        linear->projected = {};
    }
    if (start >= 0) cache_pos = start;
    speculative_start = -1;
    speculative_confirmed = 0;
}

std::vector<mfq_tensor_backend::Tensor*> State::graph_state() {
    std::vector<mfq_tensor_backend::Tensor*> result;
    for (auto& state : layers) {
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

TextSessionStateKind State::text_session_state_kind() const {
    if (full_layout(*this)) return TextSessionStateKind::FullAttention;
    if (hybrid_layout(*this)) return TextSessionStateKind::HybridAttention;
    return TextSessionStateKind::Unsupported;
}

bool State::supports_text_session_state() const {
    return text_session_state_kind() != TextSessionStateKind::Unsupported;
}

bool State::supports_paged_text_session_state() const {
    if (!full_layout(*this)) return false;
    return std::all_of(layout_.begin(), layout_.end(), [](const Layout& layer) {
        return !std::get<FullLayout>(layer.attention).sliding;
    });
}

TextSessionState State::capture_text_session_state(
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
        auto& saved_layers = std::get<std::vector<KvSnapshot>>(state.payload);
        saved_layers.reserve(layout_.size());
        for (std::size_t index = 0; index < layout_.size(); ++index) {
            const auto& layout = layout_[index];
            const auto& full = std::get<FullLayout>(layout.attention);
            const auto& cache = std::get<KVCache>(layers[index]);
            validate_cache_configuration(full, cache);
            MfqCudaGuard guard(layout.device);
            auto saved = capture_kv(cache, cache_pos, state.bytes);
            validate_full_snapshot(layout, full, saved);
            validate_full_snapshot_length(saved, cache_pos);
            saved_layers.push_back(std::move(saved));
        }
        return state;
    }

    state.payload = std::vector<HybridBlockSessionState>{};
    auto& saved_layers = std::get<std::vector<HybridBlockSessionState>>(state.payload);
    saved_layers.reserve(layout_.size());
    for (std::size_t index = 0; index < layout_.size(); ++index) {
        const auto& layout = layout_[index];
        MfqCudaGuard guard(layout.device);
        HybridBlockSessionState saved;
        if (const auto* full = std::get_if<FullLayout>(&layout.attention)) {
            const auto& cache = std::get<KVCache>(layers[index]);
            validate_cache_configuration(*full, cache);
            saved.kind = HybridBlockSessionStateKind::FullAttention;
            saved.full_attention = capture_kv(cache, cache_pos, state.bytes);
            validate_full_snapshot(layout, *full, saved.full_attention);
            validate_full_snapshot_length(saved.full_attention, cache_pos);
        } else {
            const auto& linear = std::get<LinearState>(layers[index]);
            require_state(!linear.mixer.pending() && !projected(linear.projected),
                          "Qwen recurrent session state has a pending speculative pass");
            const auto snapshot = linear.mixer.snapshot();
            validate_linear_snapshot(layout, std::get<LinearLayout>(layout.attention), snapshot,
                                     !layout.cpu && *transposed_);
            saved.kind = HybridBlockSessionStateKind::Recurrent;
            saved.mixer = snapshot;
            state.bytes += session_tensor_bytes(snapshot.convolution);
            state.bytes += session_tensor_bytes(snapshot.recurrent);
        }
        saved_layers.push_back(std::move(saved));
    }
    return state;
}

State::Restore State::prepare(const TextSessionState& state) const {
    const auto kind = text_session_state_kind();
    if (kind == TextSessionStateKind::Unsupported || state.kind() != kind ||
        state.cache_pos <= 0 || state.cache_pos > max_position_embeddings_ ||
        state.tokens.size() != static_cast<std::size_t>(state.cache_pos) ||
        speculative_start >= 0 || speculative_confirmed != 0)
        throw CudaSessionStateError("Qwen text session state is incompatible");

    const std::vector<KvSnapshot>* full_layers = nullptr;
    const std::vector<HybridBlockSessionState>* hybrid_layers = nullptr;
    if (kind == TextSessionStateKind::FullAttention) {
        full_layers = std::get_if<std::vector<KvSnapshot>>(&state.payload);
        if (full_layers == nullptr || full_layers->size() != layout_.size())
            throw CudaSessionStateError("full-attention text session state is incompatible");
        for (std::size_t index = 0; index < layout_.size(); ++index) {
            const auto& layout = layout_[index];
            const auto& full = std::get<FullLayout>(layout.attention);
            validate_cache_configuration(full, std::get<KVCache>(layers[index]));
            validate_full_snapshot(layout, full, full_layers->at(index));
            validate_full_snapshot_length(full_layers->at(index), state.cache_pos);
        }
    } else {
        hybrid_layers = std::get_if<std::vector<HybridBlockSessionState>>(&state.payload);
        if (hybrid_layers == nullptr || hybrid_layers->size() != layout_.size())
            throw CudaSessionStateError("Qwen hybrid session state is incompatible");
        for (std::size_t index = 0; index < layout_.size(); ++index) {
            const auto& layout = layout_[index];
            const auto& saved = hybrid_layers->at(index);
            if (const auto* full = std::get_if<FullLayout>(&layout.attention)) {
                if (saved.kind != HybridBlockSessionStateKind::FullAttention)
                    throw CudaSessionStateError("Qwen hybrid full-attention layer changed");
                validate_cache_configuration(*full, std::get<KVCache>(layers[index]));
                validate_full_snapshot(layout, *full, saved.full_attention);
                validate_full_snapshot_length(saved.full_attention, state.cache_pos);
            } else {
                if (saved.kind != HybridBlockSessionStateKind::Recurrent)
                    throw CudaSessionStateError("Qwen hybrid recurrent layer changed");
                validate_linear_snapshot(layout, std::get<LinearLayout>(layout.attention),
                                         saved.mixer, !layout.cpu && *transposed_);
            }
        }
    }

    Restore result;
    initialize_layers(result.layers, layout_);
    result.cache_pos = state.cache_pos;
    result.decode_position_delta = state.decode_position_delta;
    return result;
}

void State::stage(Restore& restore, const TextSessionState& state) const {
    const auto kind = text_session_state_kind();
    const auto* full_layers = kind == TextSessionStateKind::FullAttention
                                  ? std::get_if<std::vector<KvSnapshot>>(&state.payload)
                                  : nullptr;
    const auto* hybrid_layers = kind == TextSessionStateKind::HybridAttention
                                    ? std::get_if<std::vector<HybridBlockSessionState>>(
                                          &state.payload)
                                    : nullptr;
    for (std::size_t index = 0; index < layout_.size(); ++index) {
        const auto& layout = layout_[index];
        MfqCudaGuard guard(layout.device);
        if (const auto* full = std::get_if<FullLayout>(&layout.attention)) {
            const auto& saved = full_layers != nullptr
                                    ? full_layers->at(index)
                                    : hybrid_layers->at(index).full_attention;
            auto& cache = std::get<KVCache>(restore.layers[index]);
            restore_kv(cache, saved, full->head_dim, full->dtype, full->quantization);
        } else {
            const auto& saved = hybrid_layers->at(index);
            const auto& current = std::get<LinearState>(layers[index]);
            auto& linear = std::get<LinearState>(restore.layers[index]);
            linear.speculative_ffn_batches = current.speculative_ffn_batches;
            linear.speculative_projection_batches = current.speculative_projection_batches;
            // Candidate owns copy targets through the caller's asynchronous drain.
            linear.mixer.view().convolution = mfq_tensor_backend::empty(
                saved.mixer.convolution.sizes(), saved.mixer.convolution.options());
            linear.mixer.view().recurrent = mfq_tensor_backend::empty(
                saved.mixer.recurrent.sizes(), saved.mixer.recurrent.options());
            linear.mixer.restore(saved.mixer);
        }
    }
}

void State::publish(Restore&& restore) noexcept {
    layers.swap(restore.layers);
    cache_pos = restore.cache_pos;
    decode_position_delta = restore.decode_position_delta;
    speculative_start = -1;
    speculative_confirmed = 0;
    batch = 1;
}

} // namespace mfq::cuda::qwen35
