#include "engine/state/paged_kv.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mfq::cuda {

PagedKv::PagedKv(const std::vector<Layout>& layouts, int64_t context,
                 int32_t sequences, bool preallocate)
    : maximum_sequences_(sequences),
      logical_pages_per_sequence_((context + kPageSize - 1) / kPageSize),
      maximum_physical_pages_(checked_maximum_pages(sequences, logical_pages_per_sequence_)),
      allocator_(maximum_physical_pages_),
      host_page_table_(size_t(sequences) * size_t(logical_pages_per_sequence_), -1) {
    TYLOINFER_RUNTIME_CHECK(maximum_sequences_ > 0 &&
        logical_pages_per_sequence_ > 0,
        "Paged KV requires positive sequence and context limits");
    const int64_t maximum_chunks =
        (maximum_physical_pages_ + kPagesPerChunk - 1) /
        kPagesPerChunk;
    for (const auto& layout : layouts) {
        const auto heads = layout.heads;
        const auto head_dim = layout.head_dim;
        TYLOINFER_RUNTIME_CHECK(heads > 0 && head_dim > 0,
            "Paged KV encountered invalid full-attention geometry");
        const int device = layout.device;
        MfqCudaGuard guard(device);
        auto table = device_page_tables_.find(device);
        if (table == device_page_tables_.end()) {
            auto page_table = mfq_tensor_backend::empty(
                {maximum_sequences_, logical_pages_per_sequence_},
                mfq_tensor_backend::TensorOptions()
                    .device(mfq_tensor_backend::Device(
                        mfq_tensor_backend::kCUDA, device))
                    .dtype(mfq_tensor_backend::kInt32));
            page_table.fill_(-1);
            device_page_tables_.emplace(device, std::move(page_table));
        }
        auto pointer_options = mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA, device))
            .dtype(mfq_tensor_backend::kInt64);
        Pool pool;
        pool.layout = layout;
        pool.key_width = layout.quantization.enabled()
            ? cuda_kv_packed_width(int(head_dim), layout.quantization.key_bits()) : head_dim;
        pool.value_width = layout.quantization.enabled()
            ? cuda_kv_packed_width(int(head_dim), layout.quantization.value_bits()) : head_dim;
        pool.dtype = layout.quantization.enabled()
            ? mfq_tensor_backend::kInt32 : mfq_tensor_backend::kFloat16;
        pool.k_chunk_ptrs = mfq_tensor_backend::zeros(
            {maximum_chunks}, pointer_options);
        pool.v_chunk_ptrs = mfq_tensor_backend::zeros(
            {maximum_chunks}, pointer_options);
        layers_.push_back(std::move(pool));
    }
    TYLOINFER_RUNTIME_CHECK(!layers_.empty(),
        "Paged KV requires at least one full-attention layer");
    if (preallocate)
        ensure_storage(static_cast<size_t>(logical_pages_per_sequence_));
}

bool PagedKv::ensure_tokens(Sequence& sequence, int64_t token_count) {
    TYLOINFER_RUNTIME_CHECK(token_count >= 0 &&
        token_count <= logical_pages_per_sequence_ * kPageSize,
        "Paged KV sequence length exceeds the configured context");
    TYLOINFER_RUNTIME_CHECK(validate(sequence),
        "Paged KV sequence handle is invalid");
    const size_t required_pages = static_cast<size_t>(
        (token_count + kPageSize - 1) / kPageSize);
    if (required_pages <= sequence.physical_pages_.size()) return false;
    const size_t previous_size = sequence.physical_pages_.size();
    const size_t additional_pages = required_pages - previous_size;
    TYLOINFER_RUNTIME_CHECK(generation_ !=
        std::numeric_limits<std::uint64_t>::max(),
        "Paged KV allocation generation exhausted");
    const size_t live_pages = allocator_.live_pages();
    TYLOINFER_RUNTIME_CHECK(additional_pages <= maximum_physical_pages_ - live_pages,
        "Paged KV physical storage exhausted");
    const size_t required_highwatermark = std::max(
        allocator_.high_watermark(), live_pages + additional_pages);
    sequence.physical_pages_.reserve(required_pages);
    sequence.page_generations_.reserve(required_pages);
    page_generations_.resize(required_highwatermark);

    auto pages = allocator_.allocate(additional_pages);
    try {
        ensure_storage(allocator_.high_watermark());
    } catch (...) {
        allocator_.release(pages);
        publish_allocator_metrics();
        throw;
    }
    const auto generation = generation_++;
    for (const auto page : pages)
        page_generations_[static_cast<size_t>(page)] = generation;
    sequence.physical_pages_.insert(
        sequence.physical_pages_.end(), pages.begin(), pages.end());
    sequence.page_generations_.insert(
        sequence.page_generations_.end(), additional_pages, generation);
    sequence.owner_ = identity_;
    publish_allocator_metrics();
    return true;
}

void PagedKv::release(Sequence& sequence) {
    TYLOINFER_RUNTIME_CHECK(validate(sequence),
        "Paged KV sequence handle is invalid");
    if (sequence.physical_pages_.empty()) return;
    allocator_.release(sequence.physical_pages_);
    sequence.physical_pages_.clear();
    sequence.page_generations_.clear();
    sequence.owner_.reset();
    publish_allocator_metrics();
}

void PagedKv::update(const std::vector<const Sequence*>& sequences) {
    TYLOINFER_RUNTIME_CHECK(!sequences.empty() &&
        sequences.size() <= static_cast<size_t>(maximum_sequences_),
        "Paged KV cannot update the requested batch size");
    const size_t rows = sequences.size();
    const size_t row_width =
        static_cast<size_t>(logical_pages_per_sequence_);
    for (const auto * sequence : sequences) {
        TYLOINFER_RUNTIME_CHECK(sequence != nullptr &&
            sequence->physical_pages_.size() <= row_width &&
            validate(*sequence),
            "Paged KV received an invalid sequence handle");
    }
    std::fill(
        host_page_table_.begin(),
        host_page_table_.begin() + rows * row_width, -1);
    for (size_t row = 0; row < rows; ++row) {
        const auto * sequence = sequences[row];
        for (size_t logical_page = 0;
                logical_page < sequence->physical_pages_.size();
                ++logical_page) {
            const auto physical =
                sequence->physical_pages_[logical_page];
            host_page_table_[row * row_width + logical_page] = physical;
        }
    }
    const size_t bytes = rows * row_width * sizeof(std::int32_t);
    for (auto & [device, table] : device_page_tables_) {
        MfqCudaGuard guard(device);
        const auto stream = mfq_get_current_cuda_stream();
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(
            table.data_ptr<std::int32_t>(), host_page_table_.data(),
            bytes, cudaMemcpyHostToDevice, stream));
        // The next update may rewrite this host buffer and device table.
        // Order it after all previous users on the execution stream.
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    page_table_updates_metric_.fetch_add(1, std::memory_order_relaxed);
}

KVCache PagedKv::view(size_t index, int64_t batch) const {
    TYLOINFER_RUNTIME_CHECK(batch > 0 && batch <= maximum_sequences_,
        "Paged KV view batch size is invalid");
    const auto& pool = layers_.at(index);
    auto cache = KVCache::paged_view(pool.k_chunk_ptrs, pool.v_chunk_ptrs,
        device_page_tables_.at(pool.layout.device), batch, pool.layout.heads,
        pool.layout.head_dim, kPageSize, kPagesPerChunk, mfq_tensor_backend::kFloat16);
    cache.quantization = pool.layout.quantization;
    return cache;
}

PagedKv::Stats PagedKv::stats() const noexcept {
    return {kPageSize,
        int64_t(live_pages_metric_.load(std::memory_order_relaxed)),
        int64_t(peak_live_pages_metric_.load(std::memory_order_relaxed)),
        int64_t(capacity_pages_metric_.load(std::memory_order_relaxed)),
        int64_t(reserved_bytes_metric_.load(std::memory_order_relaxed)),
        int64_t(allocation_count_metric_.load(std::memory_order_relaxed)),
        int64_t(reuse_count_metric_.load(std::memory_order_relaxed)),
        int64_t(release_count_metric_.load(std::memory_order_relaxed)),
        int64_t(page_table_updates_metric_.load(std::memory_order_relaxed))};
}

size_t PagedKv::checked_maximum_pages(int32_t maximum_sequences, int64_t pages_per_sequence) {
    if (maximum_sequences <= 0 || pages_per_sequence <= 0 ||
            static_cast<uint64_t>(maximum_sequences) >
                static_cast<uint64_t>(
                    std::numeric_limits<int32_t>::max()) /
                static_cast<uint64_t>(pages_per_sequence)) {
        throw std::invalid_argument(
            "Paged KV maximum physical page count exceeds int32");
    }
    return static_cast<size_t>(maximum_sequences) *
        static_cast<size_t>(pages_per_sequence);
}

bool PagedKv::validate(const Sequence& sequence) const noexcept {
    if (sequence.physical_pages_.empty())
        return sequence.page_generations_.empty() && !sequence.owner_;
    if (sequence.owner_.get() != identity_.get() ||
            sequence.page_generations_.size() != sequence.physical_pages_.size())
        return false;
    for (size_t index = 0; index < sequence.physical_pages_.size(); ++index) {
        const auto page = sequence.physical_pages_[index];
        if (page < 0 || static_cast<size_t>(page) >= page_generations_.size() ||
                !allocator_.owns(page) ||
                page_generations_[static_cast<size_t>(page)] !=
                    sequence.page_generations_[index])
            return false;
    }
    return true;
}

void PagedKv::publish_allocator_metrics() noexcept {
    live_pages_metric_.store(
        allocator_.live_pages(), std::memory_order_relaxed);
    peak_live_pages_metric_.store(
        allocator_.peak_live_pages(), std::memory_order_relaxed);
    allocation_count_metric_.store(
        allocator_.allocation_count(), std::memory_order_relaxed);
    reuse_count_metric_.store(
        allocator_.reuse_count(), std::memory_order_relaxed);
    release_count_metric_.store(
        allocator_.release_count(), std::memory_order_relaxed);
}

void PagedKv::ensure_storage(size_t required_pages) {
    while (storage_pages_ < required_pages) {
        const size_t chunk_pages = std::min<size_t>(
            kPagesPerChunk, maximum_physical_pages_ - storage_pages_);
        TYLOINFER_RUNTIME_CHECK(chunk_pages > 0,
            "Paged KV physical storage exhausted");
        std::vector<Chunk> pending;
        pending.reserve(layers_.size());
        for (const auto & layer : layers_) {
            MfqCudaGuard guard(layer.layout.device);
            auto options = mfq_tensor_backend::TensorOptions()
                .device(mfq_tensor_backend::Device(
                    mfq_tensor_backend::kCUDA, layer.layout.device))
                .dtype(layer.dtype);
            Chunk chunk;
            chunk.k = mfq_tensor_backend::empty(
                {static_cast<int64_t>(chunk_pages), layer.layout.heads,
                 kPageSize, layer.key_width}, options);
            chunk.v = mfq_tensor_backend::empty(
                {static_cast<int64_t>(chunk_pages), layer.layout.heads,
                 kPageSize, layer.value_width}, options);
            pending.push_back(std::move(chunk));
        }
        const size_t chunk_index = storage_pages_ / kPagesPerChunk;
        for (size_t index = 0; index < layers_.size(); ++index) {
            auto & layer = layers_[index];
            auto & chunk = pending[index];
            const int64_t k_pointer = reinterpret_cast<int64_t>(
                chunk.k.data_ptr());
            const int64_t v_pointer = reinterpret_cast<int64_t>(
                chunk.v.data_ptr());
            MfqCudaGuard guard(layer.layout.device);
            const auto stream = mfq_get_current_cuda_stream();
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(
                layer.k_chunk_ptrs.data_ptr<int64_t>() + chunk_index,
                &k_pointer, sizeof(k_pointer), cudaMemcpyHostToDevice, stream));
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(
                layer.v_chunk_ptrs.data_ptr<int64_t>() + chunk_index,
                &v_pointer, sizeof(v_pointer), cudaMemcpyHostToDevice, stream));
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        for (size_t index = 0; index < layers_.size(); ++index) {
            reserved_bytes_ += pending[index].k.nbytes() +
                pending[index].v.nbytes();
            layers_[index].k_chunks.push_back(
                std::move(pending[index].k));
            layers_[index].v_chunks.push_back(
                std::move(pending[index].v));
        }
        storage_pages_ += chunk_pages;
        capacity_pages_metric_.store(
            storage_pages_, std::memory_order_relaxed);
        reserved_bytes_metric_.store(
            reserved_bytes_, std::memory_order_relaxed);
    }
}

} // namespace mfq::cuda
