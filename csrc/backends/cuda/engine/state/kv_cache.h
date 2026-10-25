#pragma once

#include "backend.h"
#include "ops/attention/attention.h"

#include <cstdint>
#include <utility>

struct KVCache {
    mfq_tensor_backend::Tensor k;
    mfq_tensor_backend::Tensor v;
    mfq_tensor_backend::Tensor k_chunk_ptrs;
    mfq_tensor_backend::Tensor v_chunk_ptrs;
    mfq_tensor_backend::Tensor page_table;
    bool ring = false;
    mfq::cuda::KvQuantization quantization;
    int64_t paged_batch = 0;
    int64_t paged_heads = 0;
    int64_t paged_head_dim = 0;
    int64_t page_size = 0;
    int64_t pages_per_chunk = 0;
    mfq_tensor_backend::ScalarType paged_dtype = mfq_tensor_backend::kFloat16;

    KVCache() = default;
    KVCache(
        int64_t batch,
        int64_t heads,
        int64_t max_sequence,
        int64_t head_dim,
        bool use_ring = false,
        mfq_tensor_backend::Device device =
            mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA),
        mfq_tensor_backend::ScalarType dtype = mfq_tensor_backend::kFloat16,
        mfq::cuda::KvQuantization quantization = {});

    static KVCache paged_view(
        mfq_tensor_backend::Tensor key_chunks,
        mfq_tensor_backend::Tensor value_chunks,
        mfq_tensor_backend::Tensor pages,
        int64_t batch,
        int64_t heads,
        int64_t head_dim,
        int64_t tokens_per_page,
        int64_t chunk_pages,
        mfq_tensor_backend::ScalarType dtype);

    bool is_paged() const noexcept { return page_size > 0; }
    bool defined() const noexcept {
        return is_paged()
            ? k_chunk_ptrs.defined() && v_chunk_ptrs.defined() && page_table.defined()
            : k.defined() && v.defined();
    }
    int64_t batch_size() const noexcept {
        return is_paged() ? paged_batch : (k.defined() ? k.size(0) : 0);
    }
    mfq_tensor_backend::ScalarType scalar_type() const {
        return is_paged() || quantization.enabled() ? paged_dtype : k.scalar_type();
    }
    mfq::cuda::AttentionKvView view() const {
        return {{}, {}, k, v, k_chunk_ptrs, v_chunk_ptrs, page_table, quantization,
                page_size, pages_per_chunk, is_paged() ? paged_heads : k.size(1), ring,
                batch_size(), is_paged() || quantization.enabled() ? paged_head_dim : k.size(3),
                scalar_type()};
    }
};
