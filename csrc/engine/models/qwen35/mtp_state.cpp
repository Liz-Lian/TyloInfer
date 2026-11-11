#include "engine/models/qwen35/mtp_state.h"
#include "engine/models/session/snapshot.h"
#include "engine/ops.h"
#include "engine/backends/cuda/native/tensor/backend.h"

#include <algorithm>
#include <limits>

namespace tyloi::cuda::qwen35 {


MtpState::MtpState(const tyloi::models::Mtp& owner, std::vector<Layout> layout, int64_t hidden)
    : tyloi::models::MtpState(owner), layout_(std::move(layout)), kv_(layout_.size()), hidden_(hidden) {}

#ifdef TYLOINFER_NATIVE_CUDA
namespace {
void add_bytes(std::vector<std::pair<tyloi::Device, std::size_t>>& result,
               tyloi::Device device, std::size_t bytes) {
    auto found = std::find_if(result.begin(), result.end(),
        [device](const auto& value) { return value.first == device; });
    if (found == result.end()) result.emplace_back(device, bytes);
    else {
        if (found->second > std::numeric_limits<std::size_t>::max() - bytes)
            throw std::overflow_error("Qwen MTP allocation size overflow");
        found->second += bytes;
    }
}
}

std::vector<std::pair<tyloi::Device, std::size_t>> MtpState::growth(
        int64_t batch, int64_t tokens, int64_t requested_capacity) const {
    TYLOINFER_RUNTIME_CHECK(batch > 0 && tokens > 0, "invalid MTP state dimensions");
    std::vector<std::pair<tyloi::Device, std::size_t>> result;
    for (size_t index = 0; index < kv_.size(); ++index) {
        const auto& layout = layout_[index];
        TYLOINFER_RUNTIME_CHECK(tokens <= layout.capacity && position_ <= layout.capacity - tokens,
            "Qwen MTP history exceeds context capacity");
        if (kv_[index].defined()) {
            TYLOINFER_RUNTIME_CHECK(kv_[index].batch == batch,
                "MTP batch changed without resetting state");
            if (kv_[index].physical_key.size(2) >= position_ + tokens) continue;
        }
        const auto capacity = requested_capacity > 0
            ? std::min(layout.capacity, std::max(position_ + tokens, requested_capacity)) : layout.capacity;
        add_bytes(result, tyloi::Device(tyloi::kCUDA, layout.device),
            tyloi::state::kv_bytes(batch, layout.heads, capacity, layout.head_dim, false,
                                   layout.dtype, layout.quantization));
    }
    return result;
}

std::unique_ptr<tyloi::models::Restore> MtpState::prepare_restore(const MtpSessionState& snapshot) {
    struct Candidate final : tyloi::models::Restore {
        MtpState& owner;
        MtpSessionState snapshot;
        MtpState::Restore staged;
        Candidate(MtpState& value, const MtpSessionState& saved)
            : owner(value), snapshot(saved), staged(value.prepare(snapshot)) {}
        std::vector<std::pair<tyloi::Device, std::size_t>> growth() const override {
            std::vector<std::pair<tyloi::Device, std::size_t>> result;
            for (const auto& block : std::get<std::vector<KvSnapshot>>(snapshot.payload)) {
                add_bytes(result, block.k.device(), restore_prefix_bytes({}, block.k, 2, block.capacity));
                add_bytes(result, block.v.device(), restore_prefix_bytes({}, block.v, 2, block.capacity));
            }
            return result;
        }
        void stage() override { owner.stage(staged, snapshot); }
        void publish() noexcept override { owner.publish(std::move(staged)); }
    };
    return std::make_unique<Candidate>(*this, snapshot);
}
#endif

void MtpState::prepare(int64_t batch, int64_t tokens, int64_t kl_capacity) {
    TYLOINFER_RUNTIME_CHECK(batch > 0 && tokens > 0, "invalid MTP state dimensions");
    for (size_t index = 0; index < kv_.size(); ++index) {
        const auto& layout = layout_[index];
        TYLOINFER_RUNTIME_CHECK(position_ <= layout.capacity - tokens,
            "Qwen MTP history exceeds context capacity");
        if (kv_[index].defined()) {
            TYLOINFER_RUNTIME_CHECK(kv_[index].batch == batch,
                "MTP batch changed without resetting state");
            if (kv_[index].physical_key.size(2) >= position_ + tokens) continue;
        }
        MfqCudaGuard guard(layout.device);
        const auto capacity = kl_capacity > 0
            ? std::min(layout.capacity, std::max(position_ + tokens, kl_capacity)) : layout.capacity;
        auto grown = tyloi::state::allocate_kv(batch, layout.heads, capacity, layout.head_dim, false,
            ::tyloi::Device(::tyloi::kCUDA, layout.device), layout.dtype, layout.quantization);
        if (kv_[index].defined() && position_ > 0) {
            auto key_prefix = tyloi::ops::slice(grown.physical_key, 2, 0, position_);
            auto value_prefix = tyloi::ops::slice(grown.physical_value, 2, 0, position_);
            tyloi::ops::copy_into(key_prefix,
                tyloi::ops::slice(kv_[index].physical_key, 2, 0, position_));
            tyloi::ops::copy_into(value_prefix,
                tyloi::ops::slice(kv_[index].physical_value, 2, 0, position_));
        }
        kv_[index] = std::move(grown);
    }
}

void MtpState::reset(int64_t batch) {
    position_ = 0;
    for (auto& cache : kv_)
        if (!cache.defined() || cache.batch != batch) cache = {};
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

std::vector<tyloi::state::KvView> MtpState::views() const {
    std::vector<tyloi::state::KvView> result;
    result.reserve(kv_.size());
    for (const auto& cache : kv_) {
        TYLOINFER_RUNTIME_CHECK(cache.defined(), "MTP KV state is unavailable");
        result.push_back(cache);
    }
    return result;
}

MtpSessionState MtpState::capture(int64_t target_position, const ::tyloi::DenseTensor& hidden) const {
    const auto position = target_position - 1;
    if (position <= 0 || position > position_ || !hidden.defined() ||
        hidden.dim() != 3 || hidden.size(0) != 1 || hidden.size(1) != 1 || hidden.size(2) != hidden_)
        throw std::runtime_error("Qwen MTP session boundary is unavailable");
    MtpSessionState saved;
    saved.cache_pos = position;
    auto& blocks = std::get<std::vector<KvSnapshot>>(saved.payload);
    blocks.reserve(kv_.size());
    for (size_t index = 0; index < kv_.size(); ++index) {
        MfqCudaGuard guard(layout_[index].device);
        blocks.push_back(capture_kv(kv_[index], position, saved.bytes));
    }
    auto saved_hidden = hidden.clone();
    saved.bytes += session_tensor_bytes(saved_hidden);
    saved.last_target_hidden = std::make_shared<::tyloi::DenseTensor>(std::move(saved_hidden));
    return saved;
}

MtpState::Restore MtpState::prepare(const MtpSessionState& saved) const {
    const auto* blocks = std::get_if<std::vector<KvSnapshot>>(&saved.payload);
    const auto* hidden = saved.last_target_hidden.representation<::tyloi::DenseTensor>();
    if (saved.cache_pos <= 0 || blocks == nullptr || blocks->size() != kv_.size() ||
        !hidden || !hidden->defined() || hidden->dim() != 3 ||
        hidden->size(0) != 1 || hidden->size(1) != 1 || hidden->size(2) != hidden_)
        throw CudaSessionStateError("Qwen MTP session state is incompatible");
    for (size_t index = 0; index < blocks->size(); ++index) {
        if (saved.cache_pos > layout_[index].capacity)
            throw CudaSessionStateError("Qwen MTP session exceeds context capacity");
        const auto& block = (*blocks)[index];
        const auto& layout = layout_[index];
        validate_kv(block);
        const auto key_width = layout.quantization.enabled()
            ? cuda_kv_packed_width(int(layout.head_dim), layout.quantization.key_bits()) : layout.head_dim;
        const auto value_width = layout.quantization.enabled()
            ? cuda_kv_packed_width(int(layout.head_dim), layout.quantization.value_bits()) : layout.head_dim;
        const auto dtype = layout.quantization.enabled() ? ::tyloi::kInt32 : layout.dtype;
        if (block.capacity > layout.capacity || block.ring ||
            block.k.size(1) != layout.heads || block.k.size(2) != saved.cache_pos ||
            block.k.size(3) != key_width || block.v.size(3) != value_width ||
            block.k.scalar_type() != dtype || !block.k.is_cuda() || !block.v.is_cuda() ||
            block.k.get_device() != layout.device || block.v.get_device() != layout.device)
            throw CudaSessionStateError("Qwen MTP session KV topology changed");
    }
    return {std::vector<tyloi::state::KvView>(kv_.size()), saved.cache_pos};
}

void MtpState::stage(Restore& restore, const MtpSessionState& saved) const {
    const auto& blocks = std::get<std::vector<KvSnapshot>>(saved.payload);
    for (size_t index = 0; index < restore.kv.size(); ++index) {
        const auto& layout = layout_[index];
        MfqCudaGuard guard(layout.device);
#ifdef TYLOINFER_NATIVE_CUDA
        stage_kv(restore.kv[index], blocks[index], layout.head_dim,
#else
        restore_kv(restore.kv[index], blocks[index], layout.head_dim,
#endif
            layout.dtype, layout.quantization);
    }
}

void MtpState::publish(Restore&& restore) noexcept {
    kv_.swap(restore.kv);
    position_ = restore.position;
}

} // namespace tyloi::cuda::qwen35
