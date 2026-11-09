#include "engine/state/kv_view.h"
#include "engine/ops.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/kernels/mfq_cuda_kv_quantization.h"

namespace {
struct KvLayout {
    int64_t key_width, value_width;
    tyloi::ScalarType storage_dtype;
    std::size_t bytes;
};

std::size_t checked_multiply(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("KV tensor size overflow");
    return a * b;
}

KvLayout kv_layout(int64_t batch, int64_t heads, int64_t capacity, int64_t head_dim,
                   bool ring, tyloi::ScalarType dtype, tyloi::state::KvQuantization quantization) {
    if (batch <= 0 || heads <= 0 || capacity <= 0 || head_dim <= 0)
        throw std::invalid_argument("invalid KV tensor dimensions");
    for (auto dimension : {batch, heads, capacity, head_dim})
        if (static_cast<std::uint64_t>(dimension) > std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("KV tensor size overflow");
    TYLOINFER_RUNTIME_CHECK(!ring || !quantization.enabled(),"TurboQuant ring cache is unsupported");
    int64_t key_width = head_dim, value_width = head_dim;
    auto storage_dtype = dtype;
    if (quantization.enabled()) {
        if (!std::isfinite(quantization.bits) || quantization.bits < 2 || quantization.bits > 8 ||
            head_dim > std::numeric_limits<int>::max())
            throw std::invalid_argument("invalid TurboQuant KV layout");
        key_width = tyloi::cuda::cuda_kv_packed_width(int(head_dim), quantization.key_bits());
        value_width = tyloi::cuda::cuda_kv_packed_width(int(head_dim), quantization.value_bits());
        storage_dtype = tyloi::kInt32;
    }
    const auto item_size = tyloi::scalar_size(storage_dtype);
    if (item_size == 0) throw std::invalid_argument("invalid KV tensor dtype");
    const auto rows = checked_multiply(checked_multiply(static_cast<std::size_t>(batch),
        static_cast<std::size_t>(heads)), static_cast<std::size_t>(capacity));
    const auto key_bytes = checked_multiply(checked_multiply(rows, static_cast<std::size_t>(key_width)), item_size);
    const auto value_bytes = checked_multiply(checked_multiply(rows, static_cast<std::size_t>(value_width)), item_size);
    if (key_bytes > std::numeric_limits<std::size_t>::max() - value_bytes)
        throw std::overflow_error("KV tensor size overflow");
    return {key_width, value_width, storage_dtype, key_bytes + value_bytes};
}
}

std::size_t tyloi::state::kv_bytes(int64_t batch, int64_t heads, int64_t capacity,
        int64_t head_dim, bool ring, ScalarType dtype, KvQuantization quantization) {
    return kv_layout(batch, heads, capacity, head_dim, ring, dtype, quantization).bytes;
}

tyloi::state::KvView tyloi::state::allocate_kv(
        int64_t B, int64_t H, int64_t max_seq, int64_t D, bool ring,
        tyloi::Device device, tyloi::ScalarType dtype, KvQuantization quantization) {
    const auto layout = kv_layout(B, H, max_seq, D, ring, dtype, quantization);
    auto opts = ::tyloi::TensorOptions().device(device).dtype(layout.storage_dtype);
    KvView result;
    result.physical_key = tyloi::ops::zeros({B,H,max_seq,layout.key_width},opts);
    result.physical_value = tyloi::ops::zeros({B,H,max_seq,layout.value_width},opts);
    result.quantization = quantization;
    result.ring = ring;
    result.batch = B;
    result.heads = H;
    result.head_dim = D;
    result.dtype = dtype;
    return result;
}
#endif
