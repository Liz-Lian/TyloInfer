#pragma once

#include "backend.h"
#include "engine/backends/cuda/ops/attention/attention.h"

#include <cstdint>
#include <utility>

struct KVCache {
    ::tyloi::DenseTensor k;
    ::tyloi::DenseTensor v;
    ::tyloi::DenseTensor k_chunk_ptrs;
    ::tyloi::DenseTensor v_chunk_ptrs;
    ::tyloi::DenseTensor page_table;
    bool ring = false;
    tyloi::cuda::KvQuantization quantization;
    int64_t paged_batch = 0;
    int64_t paged_heads = 0;
    int64_t paged_head_dim = 0;
    int64_t page_size = 0;
    int64_t pages_per_chunk = 0;
    ::tyloi::ScalarType paged_dtype = ::tyloi::kFloat16;

    KVCache() = default;
    KVCache(
        int64_t batch,
        int64_t heads,
        int64_t max_sequence,
        int64_t head_dim,
        bool use_ring = false,
        ::tyloi::Device device =
            ::tyloi::Device(::tyloi::kCUDA),
        ::tyloi::ScalarType dtype = ::tyloi::kFloat16,
        tyloi::cuda::KvQuantization quantization = {});

    static KVCache paged_view(
        ::tyloi::DenseTensor key_chunks,
        ::tyloi::DenseTensor value_chunks,
        ::tyloi::DenseTensor pages,
        int64_t batch,
        int64_t heads,
        int64_t head_dim,
        int64_t tokens_per_page,
        int64_t chunk_pages,
        ::tyloi::ScalarType dtype);

    bool is_paged() const noexcept { return page_size > 0; }
    bool defined() const noexcept {
        return is_paged()
            ? k_chunk_ptrs.defined() && v_chunk_ptrs.defined() && page_table.defined()
            : k.defined() && v.defined();
    }
    int64_t batch_size() const noexcept {
        return is_paged() ? paged_batch : (k.defined() ? k.size(0) : 0);
    }
    ::tyloi::ScalarType scalar_type() const {
        return is_paged() || quantization.enabled() ? paged_dtype : k.scalar_type();
    }
    tyloi::cuda::AttentionKvView view() const {
#ifdef TYLOINFER_NATIVE_CUDA
        const auto owner = [](const ::tyloi::DenseTensor& value) -> std::shared_ptr<::tyloi::Tensor> {
            return value.defined() ? std::make_shared<::tyloi::DenseTensor>(value) : nullptr;
        };
        return {{}, {}, owner(k), owner(v), owner(k_chunk_ptrs), owner(v_chunk_ptrs), owner(page_table), quantization,
                page_size, pages_per_chunk, is_paged() ? paged_heads : k.size(1), ring,
                batch_size(), is_paged() || quantization.enabled() ? paged_head_dim : k.size(3),
                scalar_type()};
#else
        return {{}, {}, k, v, k_chunk_ptrs, v_chunk_ptrs, page_table, quantization,
                page_size, pages_per_chunk, is_paged() ? paged_heads : k.size(1), ring,
                batch_size(), is_paged() || quantization.enabled() ? paged_head_dim : k.size(3),
                scalar_type()};
#endif
    }
};
