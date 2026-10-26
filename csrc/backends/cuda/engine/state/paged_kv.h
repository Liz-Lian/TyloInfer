#pragma once

#include "engine/state/pages.h"
#include "engine/state/kv_cache.h"

#include <atomic>
#include <map>
#include <vector>

namespace mfq::cuda {

class PagedKv {
public:
    struct Layout {
        int device = 0;
        int64_t heads = 0;
        int64_t head_dim = 0;
        KvQuantization quantization;
    };
    struct Sequence {
        std::vector<std::int32_t> physical_pages;
    };
    struct Stats {
        int64_t page_size = 0;
        int64_t live_pages = 0;
        int64_t peak_live_pages = 0;
        int64_t capacity_pages = 0;
        int64_t reserved_bytes = 0;
        int64_t allocations = 0;
        int64_t reuses = 0;
        int64_t releases = 0;
        int64_t table_updates = 0;
    };

    PagedKv(const std::vector<Layout>& layers, int64_t context, int32_t sequences,
            bool preallocate);
    PagedKv(const PagedKv&) = delete;
    PagedKv& operator=(const PagedKv&) = delete;

    bool ensure_tokens(Sequence& sequence, int64_t tokens);
    void release(Sequence& sequence);
    void update(const std::vector<const Sequence*>& sequences);
    // Views borrow chunk backing and must retire before this store.
    KVCache view(size_t layer, int64_t batch) const;
    Stats stats() const noexcept;

private:
    static constexpr int64_t kPageSize = 16;
    static constexpr int64_t kPagesPerChunk = 64;

    struct Pool {
        Layout layout;
        int64_t key_width = 0, value_width = 0;
        mfq_tensor_backend::ScalarType dtype = mfq_tensor_backend::kFloat16;
        mfq_tensor_backend::Tensor k_chunk_ptrs, v_chunk_ptrs;
        std::vector<mfq_tensor_backend::Tensor> k_chunks, v_chunks;
    };
    struct Chunk {
        mfq_tensor_backend::Tensor k, v;
    };

    static size_t checked_maximum_pages(int32_t sequences, int64_t pages_per_sequence);
    void publish_allocator_metrics() noexcept;
    void ensure_storage(size_t required_pages);

    int32_t maximum_sequences_ = 0;
    int64_t logical_pages_per_sequence_ = 0;
    size_t maximum_physical_pages_ = 0;
    mfq::engine::PagedKvPageAllocator allocator_;
    std::vector<std::int32_t> host_page_table_;
    std::map<int, mfq_tensor_backend::Tensor> device_page_tables_;
    std::vector<Pool> layers_;
    size_t storage_pages_ = 0;
    size_t reserved_bytes_ = 0;
    std::atomic<size_t> live_pages_metric_{0}, peak_live_pages_metric_{0};
    std::atomic<size_t> capacity_pages_metric_{0}, reserved_bytes_metric_{0};
    std::atomic<size_t> allocation_count_metric_{0}, reuse_count_metric_{0};
    std::atomic<size_t> release_count_metric_{0}, page_table_updates_metric_{0};
};

} // namespace mfq::cuda
