#include "models/qwen35/mtp_state.h"

#include <algorithm>

namespace mfq::cuda::qwen35 {

namespace tb = mfq_tensor_backend;

MtpState::MtpState(std::vector<Layout> layout, int64_t hidden)
    : layout_(std::move(layout)), kv_(layout_.size()), hidden_(hidden) {}

void MtpState::prepare(int64_t batch, int64_t tokens, int64_t kl_capacity) {
    TYLOINFER_RUNTIME_CHECK(batch > 0 && tokens > 0, "invalid MTP state dimensions");
    for (size_t index = 0; index < kv_.size(); ++index) {
        const auto& layout = layout_[index];
        TYLOINFER_RUNTIME_CHECK(position_ <= layout.capacity - tokens,
            "Qwen MTP history exceeds context capacity");
        if (kv_[index].defined()) {
            TYLOINFER_RUNTIME_CHECK(kv_[index].batch_size() == batch,
                "MTP batch changed without resetting state");
            if (kv_[index].k.size(2) >= position_ + tokens) continue;
        }
        MfqCudaGuard guard(layout.device);
        const auto capacity = kl_capacity > 0
            ? std::min(layout.capacity, std::max(position_ + tokens, kl_capacity)) : layout.capacity;
        KVCache grown(batch, layout.heads, capacity, layout.head_dim, false,
            tb::Device(tb::kCUDA, layout.device), layout.dtype, layout.quantization);
        if (kv_[index].defined() && position_ > 0) {
            grown.k.narrow(2, 0, position_).copy_(kv_[index].k.narrow(2, 0, position_));
            grown.v.narrow(2, 0, position_).copy_(kv_[index].v.narrow(2, 0, position_));
        }
        kv_[index] = std::move(grown);
    }
}

void MtpState::reset(int64_t batch) {
    position_ = 0;
    for (auto& cache : kv_)
        if (!cache.defined() || cache.batch_size() != batch) cache = {};
}

void MtpState::advance(int64_t tokens) {
    TYLOINFER_RUNTIME_CHECK(tokens > 0, "MTP position must advance by consumed inputs");
    position_ += tokens;
}

void MtpState::trim(int64_t position) {
    TYLOINFER_RUNTIME_CHECK(position >= 0 && position <= position_,
        "Qwen MTP cache trim position is invalid");
    position_ = position;
}

std::vector<AttentionKvView> MtpState::views() const {
    std::vector<AttentionKvView> result;
    result.reserve(kv_.size());
    for (const auto& cache : kv_) {
        TYLOINFER_RUNTIME_CHECK(cache.defined(), "MTP KV state is unavailable");
        result.push_back(cache.view());
    }
    return result;
}

MtpSessionState MtpState::capture(int64_t target_position, const tb::Tensor& hidden) const {
    const auto position = target_position - 1;
    if (position <= 0 || position > position_ || !hidden.defined() ||
        hidden.dim() != 3 || hidden.size(0) != 1 || hidden.size(1) != 1 || hidden.size(2) != hidden_)
        throw std::runtime_error("Qwen MTP session boundary is unavailable");
    MtpSessionState saved;
    saved.cache_pos = position;
    saved.blocks.reserve(kv_.size());
    for (size_t index = 0; index < kv_.size(); ++index) {
        MfqCudaGuard guard(layout_[index].device);
        saved.blocks.push_back(capture_kv(kv_[index], position, saved.bytes));
    }
    saved.last_target_hidden = hidden.clone();
    saved.bytes += session_tensor_bytes(saved.last_target_hidden);
    return saved;
}

void MtpState::restore(const MtpSessionState& saved) {
    if (saved.cache_pos <= 0 || saved.blocks.size() != kv_.size() ||
        !saved.last_target_hidden.defined() || saved.last_target_hidden.dim() != 3 ||
        saved.last_target_hidden.size(0) != 1 || saved.last_target_hidden.size(1) != 1 ||
        saved.last_target_hidden.size(2) != hidden_)
        throw CudaSessionStateError("Qwen MTP session state is incompatible");
    for (size_t index = 0; index < saved.blocks.size(); ++index) {
        if (saved.cache_pos > layout_[index].capacity)
            throw CudaSessionStateError("Qwen MTP session exceeds context capacity");
        const auto& block = saved.blocks[index];
        const auto& layout = layout_[index];
        validate_kv(block);
        const auto key_width = layout.quantization.enabled()
            ? cuda_kv_packed_width(int(layout.head_dim), layout.quantization.key_bits()) : layout.head_dim;
        const auto value_width = layout.quantization.enabled()
            ? cuda_kv_packed_width(int(layout.head_dim), layout.quantization.value_bits()) : layout.head_dim;
        const auto dtype = layout.quantization.enabled() ? tb::kInt32 : layout.dtype;
        if (block.capacity > layout.capacity || block.ring ||
            block.k.size(1) != layout.heads || block.k.size(2) != saved.cache_pos ||
            block.k.size(3) != key_width || block.v.size(3) != value_width ||
            block.k.scalar_type() != dtype || !block.k.is_cuda() || !block.v.is_cuda() ||
            block.k.get_device() != layout.device || block.v.get_device() != layout.device)
            throw CudaSessionStateError("Qwen MTP session KV topology changed");
    }
    // Publish only after every layer has restored into independent storage.
    std::vector<KVCache> restored(kv_.size());
    for (size_t index = 0; index < restored.size(); ++index) {
        const auto& layout = layout_[index];
        MfqCudaGuard guard(layout.device);
        restore_kv(restored[index], saved.blocks[index], layout.head_dim,
            layout.dtype, layout.quantization);
    }
    kv_ = std::move(restored);
    position_ = saved.cache_pos;
}

} // namespace mfq::cuda::qwen35
