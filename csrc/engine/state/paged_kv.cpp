#include "engine/state/paged_kv.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/kernels/mfq_cuda_kv_quantization.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda {

PagedKv::PagedKv(const std::vector<Layout>& layouts, int64_t context,
                 int32_t sequences, bool preallocate)
    : maximum_sequences_(sequences),
      logical_pages_per_sequence_((context + kPageSize - 1) / kPageSize),
      maximum_physical_pages_(checked_maximum_pages(sequences, logical_pages_per_sequence_)),
      allocator_(maximum_physical_pages_),
      host_page_table_(size_t(sequences) * size_t(logical_pages_per_sequence_), -1)
      , preallocate_(preallocate)
      {
    TYLOINFER_RUNTIME_CHECK(maximum_sequences_ > 0 &&
        logical_pages_per_sequence_ > 0,
        "Paged KV requires positive sequence and context limits");
    for (const auto& layout : layouts) {
        const auto heads = layout.heads;
        const auto head_dim = layout.head_dim;
        TYLOINFER_RUNTIME_CHECK(heads > 0 && head_dim > 0,
            "Paged KV encountered invalid full-attention geometry");
        Pool pool;
        pool.layout = layout;
        pool.key_width = layout.quantization.enabled()
            ? cuda_kv_packed_width(int(head_dim), layout.quantization.key_bits()) : head_dim;
        pool.value_width = layout.quantization.enabled()
            ? cuda_kv_packed_width(int(head_dim), layout.quantization.value_bits()) : head_dim;
        pool.dtype = layout.quantization.enabled()
            ? ::tyloi::kInt32 : ::tyloi::kFloat16;
        layers_.push_back(std::move(pool));
    }
    TYLOINFER_RUNTIME_CHECK(!layers_.empty(),
        "Paged KV requires at least one full-attention layer");
}

void PagedKv::initialize() {
    if (initialized_) throw std::logic_error("Paged KV is already initialized");
    const int64_t maximum_chunks=(maximum_physical_pages_+kPagesPerChunk-1)/kPagesPerChunk;
    for (auto& pool:layers_) {
        const int device=pool.layout.device;
        MfqCudaGuard guard(device);
        auto [table,inserted]=device_page_tables_.try_emplace(device);
        if (inserted) {
            table->second=::tyloi::cuda::empty({maximum_sequences_,logical_pages_per_sequence_},
                ::tyloi::TensorOptions().device(::tyloi::Device(::tyloi::kCUDA,device)).dtype(::tyloi::kInt32));
            table->second.fill_(-1);
        }
        const auto options=::tyloi::TensorOptions().device(::tyloi::Device(::tyloi::kCUDA,device)).dtype(::tyloi::kInt64);
        pool.k_chunk_ptrs=::tyloi::cuda::empty({maximum_chunks},options);
        pool.k_chunk_ptrs.zero_();
        pool.v_chunk_ptrs=::tyloi::cuda::empty({maximum_chunks},options);
        pool.v_chunk_ptrs.zero_();
    }
    if (preallocate_) ensure_storage(static_cast<size_t>(logical_pages_per_sequence_));
    initialized_=true;
}

namespace {
using Growth = std::vector<std::pair<tyloi::Device, std::size_t>>;
std::size_t checked_mul(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max()/b) throw std::overflow_error("Paged KV growth overflow");
    return a*b;
}
void add_growth(Growth& result, int device, std::size_t bytes) {
    const tyloi::Device target(tyloi::kCUDA,device);
    auto entry=std::find_if(result.begin(),result.end(),[&](const auto& v){return v.first==target;});
    if (entry==result.end()) result.emplace_back(target,bytes);
    else {
        if (entry->second>std::numeric_limits<std::size_t>::max()-bytes) throw std::overflow_error("Paged KV growth overflow");
        entry->second+=bytes;
    }
}
std::size_t layer_pages(const PagedKv::Layout& layer,std::size_t pages) {
    if (layer.heads<=0 || layer.head_dim<=0 || layer.head_dim>std::numeric_limits<int>::max())
        throw std::invalid_argument("Paged KV encountered invalid full-attention geometry");
    const auto key=layer.quantization.enabled() ? cuda_kv_packed_width(int(layer.head_dim),layer.quantization.key_bits()) : layer.head_dim;
    const auto value=layer.quantization.enabled() ? cuda_kv_packed_width(int(layer.head_dim),layer.quantization.value_bits()) : layer.head_dim;
    const auto element=layer.quantization.enabled()?4u:2u;
    return checked_mul(checked_mul(checked_mul(pages,layer.heads),16),checked_mul(std::size_t(key+value),element));
}
}
std::vector<std::pair<tyloi::Device, std::size_t>> PagedKv::initial_growth(
        const std::vector<Layout>& layers,int64_t context,int32_t sequences,bool preallocate) {
    if (context<=0 || sequences<=0 || layers.empty()) throw std::invalid_argument("invalid Paged KV geometry");
    const auto logical=(context+kPageSize-1)/kPageSize;
    const auto maximum=checked_maximum_pages(sequences,logical);
    const auto chunks=(maximum+kPagesPerChunk-1)/kPagesPerChunk;
    Growth result;
    std::vector<int> table_devices;
    for (const auto& layer:layers) {
        if (std::find(table_devices.begin(),table_devices.end(),layer.device)==table_devices.end()) {
            table_devices.push_back(layer.device);
            add_growth(result,layer.device,checked_mul(checked_mul(sequences,logical),sizeof(std::int32_t)));
        }
        add_growth(result,layer.device,checked_mul(checked_mul(chunks,2),sizeof(std::int64_t)));
        if (preallocate) {
            const auto pages=std::min(maximum,((std::size_t(logical)+kPagesPerChunk-1)/kPagesPerChunk)*kPagesPerChunk);
            add_growth(result,layer.device,layer_pages(layer,pages));
        }
    }
    return result;
}
std::vector<std::pair<tyloi::Device, std::size_t>> PagedKv::growth(
        std::span<const std::pair<const Sequence*, int64_t>> requests) const {
    const auto live=allocator_.live_pages();
    std::size_t added=0;
    std::vector<const Sequence*> seen;
    for (const auto& [sequence,tokens]:requests) {
        TYLOINFER_RUNTIME_CHECK(sequence && validate(*sequence) && tokens>=0 &&
            tokens<=logical_pages_per_sequence_*kPageSize,"Paged KV sequence handle or length is invalid");
        TYLOINFER_RUNTIME_CHECK(std::find(seen.begin(),seen.end(),sequence)==seen.end(),"Paged KV duplicate sequence growth");
        seen.push_back(sequence);
        const auto pages=std::size_t((tokens+kPageSize-1)/kPageSize);
        if (pages>sequence->physical_pages_.size()) {
            const auto delta=pages-sequence->physical_pages_.size();
            TYLOINFER_RUNTIME_CHECK(delta<=maximum_physical_pages_-live-added,"Paged KV physical storage exhausted");
            added+=delta;
        }
    }
    const auto required=std::max(allocator_.high_watermark(),live+added);
    const auto allocated=std::min(maximum_physical_pages_,((required+kPagesPerChunk-1)/kPagesPerChunk)*kPagesPerChunk);
    Growth result;
    if (allocated>storage_pages_) for (const auto& layer:layers_)
        add_growth(result,layer.layout.device,layer_pages(layer.layout,allocated-storage_pages_));
    return result;
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

tyloi::state::KvView PagedKv::view(size_t index, int64_t batch) const {
    TYLOINFER_RUNTIME_CHECK(batch > 0 && batch <= maximum_sequences_,
        "Paged KV view batch size is invalid");
    const auto& pool = layers_.at(index);
    tyloi::state::KvView cache;
    cache.key_chunk_ptrs = pool.k_chunk_ptrs;
    cache.value_chunk_ptrs = pool.v_chunk_ptrs;
    cache.page_table = device_page_tables_.at(pool.layout.device);
    cache.batch = batch;
    cache.heads = pool.layout.heads;
    cache.head_dim = pool.layout.head_dim;
    cache.page_size = kPageSize;
    cache.pages_per_chunk = kPagesPerChunk;
    cache.dtype = ::tyloi::kFloat16;
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
        pending_chunks_.clear();
        pending_chunks_.reserve(layers_.size());
        pending_pointers_.clear();
        pending_pointers_.resize(layers_.size()*2);
        auto& pending=pending_chunks_;
        for (const auto & layer : layers_) {
            MfqCudaGuard guard(layer.layout.device);
            auto options = ::tyloi::TensorOptions()
                .device(::tyloi::Device(
                    ::tyloi::kCUDA, layer.layout.device))
                .dtype(layer.dtype);
            pending.emplace_back();
            auto& chunk=pending.back();
            chunk.k = ::tyloi::cuda::empty(
                {static_cast<int64_t>(chunk_pages), layer.layout.heads,
                 kPageSize, layer.key_width}, options);
            chunk.v = ::tyloi::cuda::empty(
                {static_cast<int64_t>(chunk_pages), layer.layout.heads,
                 kPageSize, layer.value_width}, options);
        }
        for (size_t index=0;index<pending.size();++index) {
            pending_pointers_[2*index]=reinterpret_cast<int64_t>(pending[index].k.data_ptr());
            pending_pointers_[2*index+1]=reinterpret_cast<int64_t>(pending[index].v.data_ptr());
        }
        const size_t chunk_index = storage_pages_ / kPagesPerChunk;
        for (size_t index = 0; index < layers_.size(); ++index) {
            auto & layer = layers_[index];
            MfqCudaGuard guard(layer.layout.device);
            const auto stream = mfq_get_current_cuda_stream();
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(
                layer.k_chunk_ptrs.data_ptr<int64_t>() + chunk_index,
                &pending_pointers_[2*index], sizeof(int64_t), cudaMemcpyHostToDevice, stream));
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(
                layer.v_chunk_ptrs.data_ptr<int64_t>() + chunk_index,
                &pending_pointers_[2*index+1], sizeof(int64_t), cudaMemcpyHostToDevice, stream));
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
        pending.clear();
        pending_pointers_.clear();
        storage_pages_ += chunk_pages;
        capacity_pages_metric_.store(
            storage_pages_, std::memory_order_relaxed);
        reserved_bytes_metric_.store(
            reserved_bytes_, std::memory_order_relaxed);
    }
}

} // namespace tyloi::cuda
