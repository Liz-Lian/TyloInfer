#include "engine/state/kv_cache.h"
#include "engine/state/kv_view.h"

#include <stdexcept>
#include <utility>

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
tyloi::state::KvView tyloi::state::allocate_kv(
        int64_t B, int64_t H, int64_t max_seq, int64_t D, bool ring,
        tyloi::Device device, tyloi::ScalarType dtype, KvQuantization quantization) {
    auto opts = ::tyloi::TensorOptions().device(device).dtype(dtype);
    if(quantization.enabled())opts=opts.dtype(::tyloi::kInt32);
    TYLOINFER_RUNTIME_CHECK(!ring || !quantization.enabled(),"TurboQuant ring cache is unsupported");
    KvView result;
    result.physical_key = ::tyloi::cuda::zeros({B,H,max_seq,quantization.enabled()?tyloi::cuda::cuda_kv_packed_width(int(D),quantization.key_bits()):D},opts);
    result.physical_value = ::tyloi::cuda::zeros({B,H,max_seq,quantization.enabled()?tyloi::cuda::cuda_kv_packed_width(int(D),quantization.value_bits()):D},opts);
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
    k = std::move(allocated.physical_key);
    v = std::move(allocated.physical_value);
#else
    auto opts = ::tyloi::TensorOptions().device(device).dtype(dtype);
    if(quantization.enabled())opts=opts.dtype(::tyloi::kInt32);
    TYLOINFER_RUNTIME_CHECK(!ring || !quantization.enabled(),"TurboQuant ring cache is unsupported");
    k = ::tyloi::cuda::zeros({B,H,max_seq,quantization.enabled()?tyloi::cuda::cuda_kv_packed_width(int(D),quantization.key_bits()):D},opts);
    v = ::tyloi::cuda::zeros({B,H,max_seq,quantization.enabled()?tyloi::cuda::cuda_kv_packed_width(int(D),quantization.value_bits()):D},opts);
#endif
}

KVCache KVCache::paged_view(
        ::tyloi::Tensor key_chunks,
        ::tyloi::Tensor value_chunks,
        ::tyloi::Tensor pages,
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
