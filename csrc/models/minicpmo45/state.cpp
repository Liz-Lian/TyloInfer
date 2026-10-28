#include "models/minicpmo45/state.h"
#include "models/minicpmo45/model.h"

#include <stdexcept>
#include <utility>

namespace mfq::cuda::minicpmo45 {
namespace tb = mfq_tensor_backend;

State::State(const Model& model)
    : layers(model.layers.size()), maximum_(model.metadata.max_position_embeddings) {
    for (const auto& layer : model.layers)
        layout_.push_back({layer.device, layer.cpu, layer.attention.kv_heads,
            layer.attention.attention_head_dim, layer.attention.max_position_embeddings});
}

void State::preallocate() {
    for (size_t index = 0; index < layers.size(); ++index) {
        const auto& layout = layout_[index];
        if (layout.cpu || layers[index].defined()) continue;
        MfqCudaGuard guard(layout.device);
        layers[index] = KVCache(1, layout.heads, layout.capacity, layout.head_dim, false,
            tb::Device(tb::kCUDA, layout.device), tb::kFloat16);
    }
}

void State::reset(int64_t next_batch) {
    if (next_batch <= 0) throw std::invalid_argument("MiniCPM TTS batch must be positive");
    cache_pos = decode_position_delta = 0;
    batch = next_batch;
    for (size_t index = 0; index < layers.size(); ++index) {
        auto& cache = layers[index];
        if (cache.defined() && cache.batch_size() != batch) {
            MfqCudaGuard guard(layout_[index].device);
            cache = {};
        }
    }
}

std::vector<Tensor> State::storage() const {
    std::vector<Tensor> result;
    for (const auto& cache : layers) {
        if (cache.k.defined()) result.push_back(cache.k);
        if (cache.v.defined()) result.push_back(cache.v);
    }
    return result;
}

TextSessionStateKind State::text_session_state_kind() const {
    return layers.empty() ? TextSessionStateKind::Unsupported : TextSessionStateKind::FullAttention;
}

bool State::supports_text_session_state() const { return !layers.empty(); }
bool State::supports_paged_text_session_state() const { return !layers.empty(); }

TextSessionState State::capture_text_session_state(const std::vector<int64_t>& tokens) const {
    if (layers.empty() || batch != 1 || cache_pos <= 0 || size_t(cache_pos) != tokens.size())
        throw std::runtime_error("MiniCPM TTS session token count does not match the cache");
    TextSessionState state;
    state.tokens = tokens;
    state.cache_pos = cache_pos;
    state.decode_position_delta = decode_position_delta;
    state.payload = std::vector<KvSnapshot>{};
    auto& saved = std::get<std::vector<KvSnapshot>>(state.payload);
    for (size_t index = 0; index < layers.size(); ++index) {
        MfqCudaGuard guard(layout_[index].device);
        saved.push_back(capture_kv(layers[index], cache_pos, state.bytes));
    }
    return state;
}

State::Restore State::prepare(const TextSessionState& state) const {
    const auto* saved = std::get_if<std::vector<KvSnapshot>>(&state.payload);
    if (layers.empty() || !saved || saved->size() != layers.size() || state.cache_pos <= 0 ||
        state.cache_pos > maximum_ || size_t(state.cache_pos) != state.tokens.size())
        throw CudaSessionStateError("MiniCPM TTS session state is incompatible");
    for (size_t index = 0; index < saved->size(); ++index) {
        const auto& layout = layout_[index];
        const auto& cache = saved->at(index);
        validate_kv(cache);
        const auto device = tb::Device(layout.cpu ? tb::kCPU : tb::kCUDA,
                                       layout.cpu ? -1 : layout.device);
        if (cache.ring || cache.k.device() != device || cache.v.device() != device ||
            cache.k.scalar_type() != tb::kFloat16 || cache.v.scalar_type() != tb::kFloat16 ||
            cache.k.size(1) != layout.heads || cache.v.size(1) != layout.heads ||
            cache.k.size(3) != layout.head_dim || cache.v.size(3) != layout.head_dim ||
            cache.k.size(2) != state.cache_pos || cache.capacity < state.cache_pos)
            throw CudaSessionStateError("MiniCPM TTS session cache geometry changed");
    }
    return {std::vector<KVCache>(layers.size()), state.cache_pos, state.decode_position_delta};
}

void State::stage(Restore& restore, const TextSessionState& state) const {
    const auto& saved = std::get<std::vector<KvSnapshot>>(state.payload);
    for (size_t index = 0; index < layers.size(); ++index) {
        MfqCudaGuard guard(layout_[index].device);
        restore_kv(restore.layers[index], saved[index], layout_[index].head_dim, tb::kFloat16, {});
    }
}

void State::publish(Restore&& restore) noexcept {
    layers.swap(restore.layers);
    cache_pos = restore.cache_pos;
    decode_position_delta = restore.decode_position_delta;
    batch = 1;
}

} // namespace mfq::cuda::minicpmo45
