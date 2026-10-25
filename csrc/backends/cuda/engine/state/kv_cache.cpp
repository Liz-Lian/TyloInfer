#include "backends/cuda/engine/state/kv_cache.h"

#include <stdexcept>
#include <utility>

KVCache::KVCache(
        int64_t B,
        int64_t H,
        int64_t max_seq,
        int64_t D,
        bool use_ring,
        mfq_tensor_backend::Device device,
        mfq_tensor_backend::ScalarType dtype, mfq::cuda::KvQuantization settings)
    : ring(use_ring), quantization(settings), paged_batch(B), paged_heads(H),
      paged_head_dim(D), paged_dtype(dtype) {
    auto opts = mfq_tensor_backend::TensorOptions().device(device).dtype(dtype);
    if(quantization.enabled())opts=opts.dtype(mfq_tensor_backend::kInt32);
    TYLOINFER_RUNTIME_CHECK(!ring || !quantization.enabled(),"TurboQuant ring cache is unsupported");
    k = mfq_tensor_backend::zeros({B,H,max_seq,quantization.enabled()?mfq::cuda::cuda_kv_packed_width(int(D),quantization.key_bits()):D},opts);
    v = mfq_tensor_backend::zeros({B,H,max_seq,quantization.enabled()?mfq::cuda::cuda_kv_packed_width(int(D),quantization.value_bits()):D},opts);
}

KVCache KVCache::paged_view(
        mfq_tensor_backend::Tensor key_chunks,
        mfq_tensor_backend::Tensor value_chunks,
        mfq_tensor_backend::Tensor pages,
        int64_t batch, int64_t heads, int64_t head_dim,
        int64_t tokens_per_page, int64_t chunk_pages,
        mfq_tensor_backend::ScalarType dtype) {
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
