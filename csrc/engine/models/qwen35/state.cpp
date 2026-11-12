#include "engine/models/qwen35/state.h"
#include "engine/models/session/snapshot.h"

#include "engine/models/qwen35/model.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/models/qwen35/mtp_state.h"
#endif
#include "engine/models/input.h"
#include "engine/ops.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda::qwen35 {

namespace {

using DenseTensor = ::tyloi::DenseTensor;

void require_state(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

#ifndef TYLOINFER_NATIVE_CUDA
bool projected(const LinearInputs& value) {
    return value.qk.defined() || value.value.defined() || value.qkv.defined() ||
           value.gate.defined() || value.beta.defined() || value.batch != 0 ||
           value.tokens != 0 || value.split;
}
#endif

bool on_layer_device(const ::tyloi::Tensor& tensor, const Layout& layer) {
    return layer.cpu ? !tensor.is_cuda()
                     : tensor.is_cuda() && tensor.get_device() == layer.device;
}

bool full_layout(const State& state) {
    if (state.layers.empty() || state.layers.size() != state.layout().size()) return false;
    for (std::size_t index = 0; index < state.layers.size(); ++index) {
        if (!std::holds_alternative<FullLayout>(state.layout()[index].attention) ||
            !std::holds_alternative<tyloi::state::KvView>(state.layers[index]))
            return false;
    }
    return true;
}

bool hybrid_layout(const State& state) {
    if (state.layers.empty() || state.layers.size() != state.layout().size()) return false;
    for (std::size_t index = 0; index < state.layers.size(); ++index) {
        const bool full = std::holds_alternative<FullLayout>(state.layout()[index].attention) &&
                          std::holds_alternative<tyloi::state::KvView>(state.layers[index]);
        const bool linear = std::holds_alternative<LinearLayout>(state.layout()[index].attention) &&
                            std::holds_alternative<LinearState>(state.layers[index]);
        if (!full && !linear) return false;
    }
    return true;
}

void validate_linear_snapshot(const Layout& model_layer, const LinearLayout& layer,
#ifdef TYLOINFER_NATIVE_CUDA
                             const tyloi::ops::Linear::State::Snapshot& mixer, bool transposed) {
#else
                             const Recurrent::State& mixer, bool transposed) {
#endif
    const auto width = 2 * layer.key_heads * layer.key_dim +
                       layer.value_heads * layer.value_dim;
    if (!(mixer.convolution.defined() && mixer.recurrent.defined() && mixer.transposed == transposed &&
          mixer.convolution.scalar_type() == ::tyloi::kFloat32 &&
          mixer.recurrent.scalar_type() == ::tyloi::kFloat32 &&
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
    const auto dtype = quantized ? ::tyloi::kInt32 : layer.dtype;
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

void validate_cache_configuration(const FullLayout& layer, const tyloi::state::KvView& cache) {
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
#ifdef TYLOINFER_NATIVE_CUDA
      model_(&model), transposed_(transposed) {
#else
      transposed_(&transposed) {
#endif
    layout_.reserve(model.layers.size());
    for (const auto& layer : model.layers) {
        if (const auto* full = std::get_if<Full>(&layer.weights)) {
            layout_.push_back({layer.device,
                               layer.cpu,
                               FullLayout{full->attention.kv_heads,
                                          full->attention.attention_head_dim,
                                          full->attention.sliding,
                                          full->attention.official_bf16
                                              ? ::tyloi::kBFloat16
                                              : ::tyloi::kFloat16,
                                          full->attention.kv_quantization,
                                          full->attention.sliding
                                              ? full->attention.attention_window
                                              : full->attention.max_position_embeddings}});
        } else {
            const auto* linear = std::get_if<Linear>(&layer.weights);
            require_state(linear != nullptr, "Qwen3.5 layer weight kind mismatch");
#ifdef TYLOINFER_NATIVE_CUDA
            const auto& geometry = linear->geometry;
#else
            const auto& geometry = linear->attention;
#endif
            layout_.push_back({layer.device,
                               layer.cpu,
                               LinearLayout{geometry.key_heads,
                                            geometry.key_dim,
                                            geometry.value_heads,
                                            geometry.value_dim,
                                            geometry.conv_kernel}});
        }
    }
    initialize_layers(layers, layout_);
}

void State::preallocate() {
#ifdef TYLOINFER_NATIVE_CUDA
    MtpState* mtp_state = nullptr;
    if (predictor) {
        const auto* owner = model_->mtp();
        mtp_state = dynamic_cast<MtpState*>(predictor.get());
        require_state(owner && mtp_state && mtp_state->belongs_to(*owner),
                      "Qwen3.5 predictor state belongs to another model");
    }
#endif
    for (std::size_t index = 0; index < layout_.size(); ++index) {
        const auto& layer = layout_[index];
        const auto* full = std::get_if<FullLayout>(&layer.attention);
        if (layer.cpu || full == nullptr) continue;
        auto& cache = std::get<tyloi::state::KvView>(layers[index]);
        if (cache.defined()) continue;
        MfqCudaGuard guard(layer.device);
        cache = tyloi::state::allocate_kv(
            1, full->kv_heads, full->capacity, full->head_dim, full->sliding,
            ::tyloi::Device(::tyloi::kCUDA, layer.device), full->dtype,
            full->quantization);
    }
#ifdef TYLOINFER_NATIVE_CUDA
    if (mtp_state) mtp_state->prepare(1, 1, 0);
#endif
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
        if (auto* cache = std::get_if<tyloi::state::KvView>(&state)) {
            if (cache->defined() && cache->batch != new_batch) *cache = {};
            continue;
        }
        auto& linear = std::get<LinearState>(state);
#ifdef TYLOINFER_NATIVE_CUDA
        linear.attention.reset(new_batch);
#else
        linear.mixer.reset(new_batch);
        linear.projected = {};
#endif
    }
}

void State::commit_speculative() {
    require_state(speculative_start >= 0, "no speculative transaction");
    for (std::size_t index = 0; index < layers.size(); ++index) {
        MfqCudaGuard guard(layout_[index].device);
        if (auto* linear = std::get_if<LinearState>(&layers[index])) {
#ifdef TYLOINFER_NATIVE_CUDA
            linear->attention.commit();
#else
            linear->projected = {};
            linear->mixer.commit();
#endif
        }
    }
    speculative_start = -1;
    speculative_confirmed = 0;
}

std::int64_t State::rewind_suffix(std::int64_t accepted_suffix) {
    require_state(speculative_start >= 0 && accepted_suffix >= 0,
                  "no speculative transaction");
    const auto keep = speculative_start + speculative_confirmed + accepted_suffix;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        auto* linear = std::get_if<LinearState>(&layers[index]);
        if (linear == nullptr) continue;
        MfqCudaGuard guard(layout_[index].device);
#ifdef TYLOINFER_NATIVE_CUDA
        linear->attention.rewind(keep);
#else
        linear->mixer.rewind(keep);
#endif
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
#ifdef TYLOINFER_NATIVE_CUDA
            linear->attention.abort();
#else
            linear->mixer.abort();
#endif
        } catch (...) {
        }
#ifndef TYLOINFER_NATIVE_CUDA
        linear->projected = {};
#endif
    }
    if (start >= 0) cache_pos = start;
    speculative_start = -1;
    speculative_confirmed = 0;
}

std::vector<::tyloi::Tensor*> State::graph_state() {
    std::vector<::tyloi::Tensor*> result;
    for (auto& state : layers) {
        auto* linear = std::get_if<LinearState>(&state);
        if (linear == nullptr) continue;
#ifdef TYLOINFER_NATIVE_CUDA
        require_state(linear->attention.defined() && !linear->attention.pending() &&
                          !linear->attention.has_tape(),
                      "decode graph requires confirmed recurrent state");
        auto members = linear->attention.graph_state();
        result.insert(result.end(), members.begin(), members.end());
#else
        require_state(linear->mixer.defined() && !linear->mixer.pending() &&
                          !projected(linear->projected),
                      "decode graph requires confirmed recurrent state");
        result.push_back(&linear->mixer.view().convolution);
        result.push_back(&linear->mixer.view().recurrent);
#endif
    }
    return result;
}

std::vector<::tyloi::Tensor> State::storage() const {
    std::vector<::tyloi::Tensor> result;
    const auto append = [&result](const ::tyloi::Tensor& tensor) {
        if (tensor && tensor.defined()) result.push_back(tensor);
    };
    for (const auto& state : layers) {
        if (const auto* cache = std::get_if<tyloi::state::KvView>(&state)) {
            if (cache->paged()) {
                append(cache->key_chunk_ptrs);
                append(cache->value_chunk_ptrs);
                append(cache->page_table);
            } else {
                append(cache->physical_key);
                append(cache->physical_value);
            }
        } else {
            const auto& linear = std::get<LinearState>(state);
#ifdef TYLOINFER_NATIVE_CUDA
            auto members = linear.attention.storage();
            result.insert(result.end(), members.begin(), members.end());
#else
            append(linear.mixer.view().convolution);
            append(linear.mixer.view().recurrent);
#endif
        }
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

#ifdef TYLOINFER_NATIVE_CUDA
std::optional<tyloi::engine::Metrics> State::metrics() const {
    tyloi::engine::Metrics result;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        if (const auto* linear = std::get_if<LinearState>(&layers[index])) {
            const auto prefix = "linear." + std::to_string(index) + ".";
            result.emplace_back(prefix + "ffn_batches", linear->speculative_ffn_batches);
            result.emplace_back(prefix + "projection_batches", linear->speculative_projection_batches);
        }
    }
    return result;
}
#endif

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
            const auto& cache = std::get<tyloi::state::KvView>(layers[index]);
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
            const auto& cache = std::get<tyloi::state::KvView>(layers[index]);
            validate_cache_configuration(*full, cache);
            saved.kind = HybridBlockSessionStateKind::FullAttention;
            saved.full_attention = capture_kv(cache, cache_pos, state.bytes);
            validate_full_snapshot(layout, *full, saved.full_attention);
            validate_full_snapshot_length(saved.full_attention, cache_pos);
        } else {
            const auto& linear = std::get<LinearState>(layers[index]);
#ifdef TYLOINFER_NATIVE_CUDA
            require_state(!linear.attention.pending() && !linear.attention.has_tape(),
                          "Qwen recurrent session state has a pending speculative pass");
            const auto snapshot = linear.attention.snapshot();
#else
            require_state(!linear.mixer.pending() && !projected(linear.projected),
                          "Qwen recurrent session state has a pending speculative pass");
            const auto snapshot = linear.mixer.snapshot();
#endif
            validate_linear_snapshot(layout, std::get<LinearLayout>(layout.attention), snapshot,
#ifdef TYLOINFER_NATIVE_CUDA
                                     !layout.cpu && transposed_);
#else
                                     !layout.cpu && *transposed_);
#endif
            saved.kind = HybridBlockSessionStateKind::Recurrent;
            saved.convolution = snapshot.convolution;
            saved.recurrent = snapshot.recurrent;
            saved.transposed = snapshot.transposed;
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
            validate_cache_configuration(full, std::get<tyloi::state::KvView>(layers[index]));
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
                validate_cache_configuration(*full, std::get<tyloi::state::KvView>(layers[index]));
                validate_full_snapshot(layout, *full, saved.full_attention);
                validate_full_snapshot_length(saved.full_attention, state.cache_pos);
            } else {
                if (saved.kind != HybridBlockSessionStateKind::Recurrent)
                    throw CudaSessionStateError("Qwen hybrid recurrent layer changed");
                if (!saved.convolution.representation<DenseTensor>() ||
                    !saved.recurrent.representation<DenseTensor>())
                    throw CudaSessionStateError("Qwen recurrent session topology changed");
                validate_linear_snapshot(layout, std::get<LinearLayout>(layout.attention),
#ifdef TYLOINFER_NATIVE_CUDA
                                         tyloi::ops::Linear::State::Snapshot{saved.convolution, saved.recurrent, saved.transposed},
                                         !layout.cpu && transposed_);
#else
                                         Recurrent::State{saved.convolution, saved.recurrent, saved.transposed},
                                         !layout.cpu && *transposed_);
#endif
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
            auto& cache = std::get<tyloi::state::KvView>(restore.layers[index]);
#ifdef TYLOINFER_NATIVE_CUDA
            stage_kv(cache, saved, full->head_dim, full->dtype, full->quantization);
#else
            restore_kv(cache, saved, full->head_dim, full->dtype, full->quantization);
#endif
        } else {
            const auto& saved = hybrid_layers->at(index);
            const auto& current = std::get<LinearState>(layers[index]);
            auto& linear = std::get<LinearState>(restore.layers[index]);
            linear.speculative_ffn_batches = current.speculative_ffn_batches;
            linear.speculative_projection_batches = current.speculative_projection_batches;
            const auto& convolution = saved.convolution;
            const auto& recurrent = saved.recurrent;
#ifdef TYLOINFER_NATIVE_CUDA
            linear.attention.restore({convolution, recurrent, saved.transposed});
#else
            // Candidate owns copy targets through the caller's asynchronous drain.
            linear.mixer.view().convolution = ::tyloi::ops::empty(
                convolution.dense().sizes(), convolution.options());
            linear.mixer.view().recurrent = ::tyloi::ops::empty(
                recurrent.dense().sizes(), recurrent.options());
            linear.mixer.restore(Recurrent::State{convolution, recurrent, saved.transposed});
#endif
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

#ifdef TYLOINFER_NATIVE_CUDA
namespace {
void add_bytes(std::vector<std::pair<tyloi::Device, std::size_t>>& result,
               tyloi::Device device, std::size_t bytes) {
    if (!device.is_cuda()) return;
    auto found = std::find_if(result.begin(), result.end(),
        [device](const auto& value) { return value.first == device; });
    if (found == result.end()) result.emplace_back(device, bytes);
    else {
        if (found->second > std::numeric_limits<std::size_t>::max() - bytes)
            throw std::overflow_error("Qwen state allocation size overflow");
        found->second += bytes;
    }
}
}

std::vector<std::pair<tyloi::Device, std::size_t>> State::reset_growth(std::int64_t next_batch) const {
    require_state(next_batch > 0, "Qwen3.5 batch must be positive");
    return {};
}

std::vector<std::pair<tyloi::Device, std::size_t>> State::growth(
        std::int64_t next_batch, std::int64_t tokens,
        const tyloi::models::ForwardOptions& options, std::int64_t confirmed) const {
    require_state(next_batch > 0 && tokens > 0 && cache_pos >= 0,
                  "invalid Qwen3.5 forward geometry");
    require_state(confirmed >= 0 && confirmed < tokens, "invalid Qwen3.5 verification window");
    require_state(speculative_start < 0, "commit or roll back the pending speculative pass before forwarding");
    require_state(tokens <= max_position_embeddings_ && cache_pos <= max_position_embeddings_ - tokens,
                  "Qwen3.5 history exceeds context capacity");
    if (cache_pos > 0) require_state(batch == next_batch, "Qwen3.5 cache batch does not match input batch");
    const auto end = cache_pos + tokens;
    std::vector<std::pair<tyloi::Device, std::size_t>> result;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        const auto& layout = layout_[index];
        const auto device = layout.cpu ? tyloi::Device(tyloi::kCPU) : tyloi::Device(tyloi::kCUDA, layout.device);
        if (const auto* full = std::get_if<FullLayout>(&layout.attention)) {
            const auto& cache = std::get<tyloi::state::KvView>(layers[index]);
            if (cache.defined() && (cache_pos != 0 || cache.batch == next_batch)) continue;
            const auto capacity = full->sliding ? full->capacity :
                (options.kv_capacity > 0 ? std::max(end, options.kv_capacity) : full->capacity);
            add_bytes(result, device, tyloi::state::kv_bytes(next_batch, full->kv_heads,
                capacity, full->head_dim, full->sliding, full->dtype, full->quantization));
        } else {
            const auto& layer = std::get<Linear>(model_->layers[index].weights);
            const auto& state = std::get<LinearState>(layers[index]);
            require_state(bool(layer.attention), "Qwen3.5 recurrent layer weight kind mismatch");
            add_bytes(result, device, layer.attention->growth(state.attention, next_batch, tokens,
                                                              confirmed, cache_pos == 0));
        }
    }
    return result;
}

void State::rollback_speculative(std::int64_t accepted) {
    require_state(speculative_start >= 0 && accepted >= 0, "no speculative transaction");
    try {
        const auto keep = rewind_suffix(accepted);
        for (std::size_t index = 0; index < layers.size(); ++index) {
            auto* linear = std::get_if<LinearState>(&layers[index]);
            if (!linear) continue;
            const auto* layer = std::get_if<Linear>(&model_->layers[index].weights);
            require_state(layer && layer->attention, "Qwen3.5 recurrent layer weight kind mismatch");
            layer->attention->replay(linear->attention, keep - speculative_start);
        }
        commit_speculative();
        cache_pos = keep;
    } catch (...) {
        abort_speculative();
        throw;
    }
}

std::unique_ptr<tyloi::models::Restore> State::prepare_restore(const TextSessionState& snapshot) {
    struct Candidate final : tyloi::models::Restore {
        State& owner;
        TextSessionState snapshot;
        State::Restore staged;
        Candidate(State& value, const TextSessionState& saved)
            : owner(value), snapshot(saved), staged(value.prepare(snapshot)) {}
        std::vector<std::pair<tyloi::Device, std::size_t>> growth() const override {
            std::vector<std::pair<tyloi::Device, std::size_t>> result;
            const auto add_kv = [&](const KvSnapshot& block) {
                add_bytes(result, block.k.device(), restore_prefix_bytes({}, block.k, 2, block.capacity));
                add_bytes(result, block.v.device(), restore_prefix_bytes({}, block.v, 2, block.capacity));
            };
            if (const auto* blocks = std::get_if<std::vector<KvSnapshot>>(&snapshot.payload)) {
                for (const auto& block : *blocks) add_kv(block);
            } else {
                for (const auto& block : std::get<std::vector<HybridBlockSessionState>>(snapshot.payload)) {
                    if (block.kind == HybridBlockSessionStateKind::FullAttention) add_kv(block.full_attention);
                    else {
                        add_bytes(result, block.convolution.device(), restore_tensor_bytes({}, block.convolution));
                        add_bytes(result, block.recurrent.device(), restore_tensor_bytes({}, block.recurrent));
                    }
                }
            }
            return result;
        }
        void stage() override { owner.stage(staged, snapshot); }
        void publish() noexcept override { owner.publish(std::move(staged)); }
    };
    return std::make_unique<Candidate>(*this, snapshot);
}
#endif

} // namespace tyloi::cuda::qwen35
