#include "engine/state/kv_cache.h"
#include "engine/state/kv_view.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
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
    result.physical_key = std::make_shared<DenseTensor>(::tyloi::cuda::zeros({B,H,max_seq,layout.key_width},opts));
    result.physical_value = std::make_shared<DenseTensor>(::tyloi::cuda::zeros({B,H,max_seq,layout.value_width},opts));
    result.quantization = quantization;
    result.ring = ring;
    result.batch = B;
    result.heads = H;
    result.head_dim = D;
    result.dtype = dtype;
    return result;
}
#endif

KVCache::KVCache(
        int64_t B,
        int64_t H,
        int64_t max_seq,
        int64_t D,
        bool use_ring,
        ::tyloi::Device device,
        ::tyloi::ScalarType dtype, tyloi::cuda::KvQuantization settings)
    : ring(use_ring), quantization(settings), paged_batch(B), paged_heads(H),
      paged_head_dim(D), paged_dtype(dtype) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    auto allocated = tyloi::state::allocate_kv(B, H, max_seq, D, use_ring, device, dtype, settings);
    k = dynamic_cast<const ::tyloi::DenseTensor&>(*allocated.physical_key);
    v = dynamic_cast<const ::tyloi::DenseTensor&>(*allocated.physical_value);
#else
    auto opts = ::tyloi::TensorOptions().device(device).dtype(dtype);
    if(quantization.enabled())opts=opts.dtype(::tyloi::kInt32);
    TYLOINFER_RUNTIME_CHECK(!ring || !quantization.enabled(),"TurboQuant ring cache is unsupported");
    k = ::tyloi::cuda::zeros({B,H,max_seq,quantization.enabled()?tyloi::cuda::cuda_kv_packed_width(int(D),quantization.key_bits()):D},opts);
    v = ::tyloi::cuda::zeros({B,H,max_seq,quantization.enabled()?tyloi::cuda::cuda_kv_packed_width(int(D),quantization.value_bits()):D},opts);
#endif
}

KVCache KVCache::paged_view(
        ::tyloi::DenseTensor key_chunks,
        ::tyloi::DenseTensor value_chunks,
        ::tyloi::DenseTensor pages,
        int64_t batch, int64_t heads, int64_t head_dim,
        int64_t tokens_per_page, int64_t chunk_pages,
        ::tyloi::ScalarType dtype) {
    KVCache result;
    result.k_chunk_ptrs = std::move(key_chunks);
    result.v_chunk_ptrs = std::move(value_chunks);
    result.page_table = std::move(pages);
    result.paged_batch = batch;
    result.paged_heads = heads;
    result.paged_head_dim = head_dim;
    result.page_size = tokens_per_page;
    result.pages_per_chunk = chunk_pages;
    result.paged_dtype = dtype;
    return result;
}
