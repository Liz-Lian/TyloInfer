#include "engine/models/gemma4/state.h"
#include "engine/models/gemma4/model.h"
#include "engine/models/common/session/paged.h"
#include "engine/backends/cuda/executor/options.h"

#include <stdexcept>

namespace mfq::cuda::gemma4 {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
std::unique_ptr<mfq::models::State> Model::make_state() const {
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
                           attention.official_bf16 ? mfq_tensor_backend::kBFloat16 : mfq_tensor_backend::kFloat16,
                           attention.kv_quantization});
    }
}

void State::preallocate() {
    if (!runtime_options::kv_vram_preallocated()) return;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        if (layers[index].defined()) continue;
        const auto& layout = layout_[index];
        MfqCudaGuard guard(layout.device);
        layers[index] = KVCache(1, layout.heads, layout.capacity, layout.head_dim, layout.sliding,
            mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, layout.device),
            layout.dtype, layout.quantization);
    }
}

void State::reset(std::int64_t next_batch) {
    cache_pos = decode_position_delta = 0;
    batch = next_batch;
    for (std::size_t index = 0; index < layers.size(); ++index) {
        auto& cache = layers[index];
        if (!cache.defined() || cache.batch_size() != next_batch) {
            MfqCudaGuard guard(layout_[index].device);
            cache = KVCache();
        }
    }
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
std::vector<mfq::Tensor> State::storage() const {
    std::vector<mfq::Tensor> result;
#else
std::vector<Tensor> State::storage() const {
    std::vector<Tensor> result;
#endif
    for (const auto& cache : layers) {
        if (cache.is_paged()) {
            if (cache.k_chunk_ptrs.defined()) result.push_back(cache.k_chunk_ptrs);
            if (cache.v_chunk_ptrs.defined()) result.push_back(cache.v_chunk_ptrs);
            if (cache.page_table.defined()) result.push_back(cache.page_table);
        } else {
            if (cache.k.defined()) result.push_back(cache.k);
            if (cache.v.defined()) result.push_back(cache.v);
        }
    }
    return result;
}

TextSessionStateKind State::text_session_state_kind() const { return TextSessionStateKind::Unsupported; }
bool State::supports_text_session_state() const { return false; }
bool State::supports_paged_text_session_state() const { return false; }
TextSessionState State::capture_text_session_state(const std::vector<std::int64_t>&) const {
    throw std::runtime_error("text session state is unsupported by this model");
}
void State::restore_text_session_state(const TextSessionState&) {
    throw CudaSessionStateError("text session state is unsupported by this model");
}

} // namespace mfq::cuda::gemma4
