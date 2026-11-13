#include "kv.h"

#include "engine/backends/cuda/kernels/mfq_cuda_cache_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_paged_kv.h"
#include "engine/backends/cuda/executor/executor.h"

#include <optional>
#include <utility>

using ::tyloi::indexing::Slice;

namespace tyloi::cuda {

std::pair<tyloi::DenseTensor, tyloi::DenseTensor> ops::append_kv(
        tyloi::state::KvView cache, tyloi::DenseTensor key, tyloi::DenseTensor value,
        tyloi::DenseTensor positions, int64_t start_pos, bool contiguous_prefill_prefix) {
    ::tyloi::DenseTensor kk(key), vv(value), pos(positions);
    const auto end_pos = start_pos + kk.size(2);
    const bool aten_write = Executor::current().attention_options().write_aten;
    std::optional<MfqCudaGuard> guard;
    if (key.is_cuda()) guard.emplace(key.device().index);
    ::tyloi::DenseTensor k = dense_kv_buffer(cache.physical_key), v = dense_kv_buffer(cache.physical_value);
    const auto k_chunk_ptrs = dense_kv_buffer(cache.key_chunk_ptrs);
    const auto v_chunk_ptrs = dense_kv_buffer(cache.value_chunk_ptrs);
    const auto page_table = dense_kv_buffer(cache.page_table);
    const auto& quantization = cache.quantization;
    const auto ring = cache.ring;
    const auto paged_batch = cache.batch;
    const auto paged_heads = cache.heads;
    const auto paged_head_dim = cache.head_dim;
    const auto page_size = cache.page_size;
    const auto pages_per_chunk = cache.pages_per_chunk;
    auto kh = kk.to(cache.dtype).contiguous();
    auto vh = vv.to(cache.dtype).contiguous();
    if(quantization.enabled()) {
        auto packed_k=tyloi::cuda::cuda_kv_encode(kh,quantization.key_bits(),false);
        auto packed_v=tyloi::cuda::cuda_kv_encode(vh,quantization.value_bits(),true);
        if(cache.paged())tyloi::cuda::cuda_kv_write_paged(quantized_pages(cache),packed_k,packed_v,pos);
        else if(pos.dim()==1) {
            k.index_copy_(2,pos,packed_k);v.index_copy_(2,pos,packed_v);
        } else {
            for(int64_t b=0;b<kh.size(0);++b) {
                auto at=pos.narrow(0,b,1).reshape({-1});
                k.narrow(0,b,1).index_copy_(2,at,packed_k.narrow(0,b,1));
                v.narrow(0,b,1).index_copy_(2,at,packed_v.narrow(0,b,1));
            }
        }
        return {k,v};
    }
    if (cache.paged()) {
        TYLOINFER_RUNTIME_CHECK(
            paged_batch == kh.size(0) && paged_heads == kh.size(1) &&
            paged_head_dim == kh.size(3) &&
            page_size > 0 && pages_per_chunk > 0,
            "Paged KV cache geometry does not match the write");
        paged_kv_cache_write_cuda(
            k_chunk_ptrs, v_chunk_ptrs, page_table,
            kh, vh, pos, page_size, pages_per_chunk);
        // Later prefill chunks must attend to the complete previous
        // prefix, including a final one-token chunk. Decode reads pages
        // directly and does not materialize a contiguous prefix.
        if (contiguous_prefill_prefix && start_pos > 0) {
            auto options = ::tyloi::TensorOptions()
                .device(kh.device()).dtype(kh.scalar_type());
            auto prefix_k = ::tyloi::cuda::empty(
                {paged_batch, paged_heads, end_pos, paged_head_dim}, options);
            auto prefix_v = ::tyloi::cuda::empty(
                {paged_batch, paged_heads, end_pos, paged_head_dim}, options);
            paged_kv_cache_gather_cuda(k_chunk_ptrs, v_chunk_ptrs,
                page_table, prefix_k, prefix_v, page_size, pages_per_chunk);
            return {prefix_k, prefix_v};
        }
        return {kh, vh};
    }
    TYLOINFER_RUNTIME_CHECK(
        kh.dim() == 4 && vh.sizes() == kh.sizes() &&
        kh.size(0) == k.size(0) && kh.size(1) == k.size(1) &&
        kh.size(3) == k.size(3) &&
        ((pos.dim() == 1 && pos.numel() == kh.size(2)) ||
         (pos.dim() == 2 && pos.size(0) == kh.size(0) &&
          pos.size(1) == kh.size(2))),
        "KV cache write requires positions [T] or [B,T]");
    if (k.is_cuda() && !aten_write && !ring) {
        auto out = kv_cache_write_cuda(k, v, kh, vh, pos);
        (void)out;
    } else {
        const int64_t batches = pos.dim() == 1 ? 1 : pos.size(0);
        for (int64_t batch = 0; batch < batches; ++batch) {
            auto cache_k = pos.dim() == 1 ? k : k.narrow(0, batch, 1);
            auto cache_v = pos.dim() == 1 ? v : v.narrow(0, batch, 1);
            auto source_k = pos.dim() == 1 ? kh : kh.narrow(0, batch, 1);
            auto source_v = pos.dim() == 1 ? vh : vh.narrow(0, batch, 1);
            auto write_pos = pos.dim() == 1
                ? pos : pos.narrow(0, batch, 1).reshape({-1});
            if (!k.is_cuda() || aten_write) {
                auto slots = ring
                    ? ::tyloi::cuda::remainder(write_pos, k.size(2))
                    : write_pos;
                slots = slots.to(::tyloi::kInt64).contiguous();
                cache_k.index_copy_(2, slots, source_k);
                cache_v.index_copy_(2, slots, source_v);
            } else {
                auto out = ring
                    ? kv_cache_write_ring_positions_cuda(
                        cache_k, cache_v, source_k, source_v, write_pos)
                    : kv_cache_write_cuda(
                        cache_k, cache_v, source_k, source_v, write_pos);
                (void)out;
            }
        }
    }
    if (ring) return {k, v};
    return {k.index({Slice(), Slice(), Slice(0, end_pos), Slice()}),
            v.index({Slice(), Slice(), Slice(0, end_pos), Slice()})};
}

} // namespace tyloi::cuda
