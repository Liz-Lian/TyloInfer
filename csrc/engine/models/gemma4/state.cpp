#include "engine/models/gemma4/state.h"
#include "engine/models/gemma4/model.h"
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/ops.h"
#else
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/models/common/session/paged.h"
#endif

#include <stdexcept>
#include <utility>

namespace tyloi::cuda::gemma4 {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
std::unique_ptr<tyloi::models::State> Model::make_state() const {
    return std::make_unique<State>(*this);
}
#endif

State::State(const Model& model) : layers(model.layers.size()) {
    layout_.reserve(model.layers.size());
    for (const auto& layer : model.layers) {
        const auto& attention = layer.attention;
        layout_.push_back({layer.device, attention.kv_heads, attention.attention_head_dim,
                           attention.sliding ? attention.attention_window : attention.max_position_embeddings,
                           attention.sliding,
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                           attention.official_bf16 ? tyloi::ScalarType::bfloat16 : tyloi::ScalarType::float16,
#else
                           attention.official_bf16 ? ::tyloi::kBFloat16 : ::tyloi::kFloat16,
#endif
                           attention.kv_quantization});
    }
}

void State::preallocate() {
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
    if (!runtime_options::kv_vram_preallocated()) return;
#endif
    for (std::size_t index = 0; index < layers.size(); ++index) {
        if (layers[index].defined()) continue;
        const auto& layout = layout_[index];
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        layers[index] = tyloi::state::allocate_kv(1, layout.heads, layout.capacity, layout.head_dim, layout.sliding,
            tyloi::Device(tyloi::DeviceType::cuda, layout.device), layout.dtype, layout.quantization);
#else
        MfqCudaGuard guard(layout.device);
        layers[index] = KVCache(1, layout.heads, layout.capacity, layout.head_dim, layout.sliding,
            ::tyloi::Device(::tyloi::kCUDA, layout.device),
            layout.dtype, layout.quantization);
#endif
    }
}

void State::reset(std::int64_t next_batch) {
    cache_pos = decode_position_delta = 0;
    batch = next_batch;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        auto& cache = layers[index];
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if (!cache.defined() || cache.batch != next_batch) {
            cache = {};
        }
#else
        if (!cache.defined() || cache.batch_size() != next_batch) {
            MfqCudaGuard guard(layout_[index].device);
            cache = KVCache();
        }
#endif
    }
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
std::vector<tyloi::Tensor*> State::graph_state() {
    if (cache_pos <= 0 || layers.empty())
        throw std::runtime_error("Gemma4 graph warmup requires a prefilled KV cache");

    std::vector<std::shared_ptr<tyloi::Tensor>> views;
    views.reserve(layers.size() * 2);
    for (const auto& cache : layers) {
        if (cache.paged() || !cache.defined() || cache.physical_key->dim() != 4 ||
            cache.physical_value->dim() != 4 || cache.physical_key->size(2) <= 0 ||
            cache.physical_value->size(2) != cache.physical_key->size(2))
            throw std::runtime_error("Gemma4 graph warmup requires allocated non-paged KV caches with matching capacities");
        const auto capacity = cache.physical_key->size(2);
        if (!cache.ring && cache_pos >= capacity)
            throw std::runtime_error("Gemma4 graph warmup exceeds KV cache capacity");
        const auto position = cache.ring ? cache_pos % capacity : cache_pos;
        views.push_back(tyloi::ops::slice(*cache.physical_key, 2, position, position + 1));
        views.push_back(tyloi::ops::slice(*cache.physical_value, 2, position, position + 1));
    }
    std::vector<tyloi::Tensor*> result;
    result.reserve(views.size());
    for (const auto& view : views) result.push_back(view.get());
    graph_views_ = std::move(views);
    return result;
}

std::vector<std::shared_ptr<const tyloi::Tensor>> State::storage() const {
    std::vector<std::shared_ptr<const tyloi::Tensor>> result;
#else
std::vector<DenseTensor> State::storage() const {
    std::vector<DenseTensor> result;
#endif
    for (const auto& cache : layers) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if (cache.paged()) {
            if (cache.key_chunk_ptrs && cache.key_chunk_ptrs->defined()) result.push_back(cache.key_chunk_ptrs);
            if (cache.value_chunk_ptrs && cache.value_chunk_ptrs->defined()) result.push_back(cache.value_chunk_ptrs);
            if (cache.page_table && cache.page_table->defined()) result.push_back(cache.page_table);
        } else {
            if (cache.physical_key && cache.physical_key->defined()) result.push_back(cache.physical_key);
            if (cache.physical_value && cache.physical_value->defined()) result.push_back(cache.physical_value);
        }
#else
        if (cache.is_paged()) {
            if (cache.k_chunk_ptrs.defined()) result.push_back(cache.k_chunk_ptrs);
            if (cache.v_chunk_ptrs.defined()) result.push_back(cache.v_chunk_ptrs);
            if (cache.page_table.defined()) result.push_back(cache.page_table);
        } else {
            if (cache.k.defined()) result.push_back(cache.k);
            if (cache.v.defined()) result.push_back(cache.v);
        }
#endif
    }
    return result;
}

bool State::supports_text_session_state() const { return false; }
bool State::supports_paged_text_session_state() const { return false; }
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
TextSessionStateKind State::text_session_state_kind() const { return TextSessionStateKind::Unsupported; }
TextSessionState State::capture_text_session_state(const std::vector<std::int64_t>&) const {
    throw std::runtime_error("text session state is unsupported by this model");
}
void State::restore_text_session_state(const TextSessionState&) {
    throw CudaSessionStateError("text session state is unsupported by this model");
}
#endif

} // namespace tyloi::cuda::gemma4
