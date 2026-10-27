#include "mlx_streamed_weight_pool.h"
#include "mlx_pinned_pool_storage.h"
#include "mlx_weight_residency.h"
#include "mlx_buffer_view.h"

#include <mlx/allocator.h>
#include <algorithm>
#include <climits>
#include <cstdio>
#include <map>
#include <stdexcept>

namespace mfq::metal {
using namespace mlx::core;

struct MlxStreamedWeightPool::Impl {
    const std::size_t page = std::size_t(getpagesize());
    struct Buffer {
        std::shared_ptr<detail::MlxPinnedPoolStorage> owner;
        allocator::Buffer value{nullptr};
        ~Buffer() { if (value.ptr()) allocator::release(value); }
    };
    std::shared_ptr<detail::MlxPinnedPoolStorage> storage;
    std::vector<array> chunks;
    std::shared_ptr<MlxWeightResidency> residency;
    std::map<std::size_t, std::size_t> free, occupied;
    std::size_t limit, chunk_bytes, capacity = 0, generation = 0;

    Impl(std::size_t requested, std::size_t maximum_entry) : limit(requested / page * page) {
        auto* device = mlx::core::metal::device(Device::gpu).mtl_device();
        const auto maximum_buffer = std::size_t(device->maxBufferLength()) / page * page;
        if (!limit || !maximum_entry || maximum_entry > maximum_buffer)
            throw std::invalid_argument("streamed weight pool cannot hold its expert geometry");
        chunk_bytes = std::min(maximum_buffer, std::max(std::size_t{512} << 20,
            (maximum_entry + page - 1) / page * page));
        storage = std::make_shared<detail::MlxPinnedPoolStorage>();
        storage->bytes = std::max(limit, std::size_t(device->recommendedMaxWorkingSetSize())) / page * page;
        storage->data = mmap(nullptr, storage->bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (storage->data == MAP_FAILED) throw std::bad_alloc();
    }
    void insert_free(std::size_t offset, std::size_t bytes) {
        while (bytes) {
            auto count = std::min(bytes, chunk_bytes - offset % chunk_bytes);
            auto begin = offset;
            auto next = free.lower_bound(begin);
            if (next != free.begin()) {
                const auto previous = std::prev(next);
                if (previous->first + previous->second == begin && previous->first / chunk_bytes == begin / chunk_bytes) {
                    begin = previous->first; count += previous->second; free.erase(previous);
                }
            }
            next = free.lower_bound(begin);
            if (next != free.end() && begin + count == next->first && begin / chunk_bytes == next->first / chunk_bytes) {
                count += next->second; free.erase(next);
            }
            free.emplace(begin, count);
            const auto consumed = std::min(bytes, chunk_bytes - offset % chunk_bytes);
            offset += consumed; bytes -= consumed;
        }
    }
    void rebuild_free() {
        free.clear();
        std::size_t begin = 0;
        for (const auto& [offset, bytes] : occupied) {
            if (offset > begin) insert_free(begin, offset - begin);
            begin = offset + bytes;
        }
        if (begin < capacity) insert_free(begin, capacity - begin);
    }
    void resize(std::size_t bytes) {
        bytes = bytes / page * page;
        if (bytes == capacity) return;
        if (bytes > storage->bytes || (!occupied.empty() && occupied.rbegin()->first + occupied.rbegin()->second > bytes))
            throw std::runtime_error("streamed weight pool resize would discard live experts");
        auto next = chunks;
        const auto count = (bytes + chunk_bytes - 1) / chunk_bytes;
        std::size_t keep = 0;
        while (keep < next.size() && keep < count &&
            next[keep].nbytes() == std::min(chunk_bytes, bytes - keep * chunk_bytes)) ++keep;
        next.erase(next.begin() + keep, next.end());
        if (bytes > capacity) madvise(static_cast<std::byte*>(storage->data) + capacity, bytes - capacity, MADV_FREE_REUSE);
        for (auto part = keep; part < count; ++part) {
            const auto length = std::min(chunk_bytes, bytes - part * chunk_bytes);
            auto owner = std::make_shared<Buffer>(); owner->owner = storage;
            owner->value = allocator::make_buffer(static_cast<std::byte*>(storage->data) + part * chunk_bytes, length);
            if (!owner->value.ptr()) throw std::runtime_error("cannot register streamed expert memory with Metal");
            next.push_back(mlx_buffer_view(owner->value, Shape{int(length / 4096), 4096}, uint8, owner));
        }
        if (residency) residency->update_arrays(next, bytes);
        else if (bytes) residency = MlxWeightResidency::retain_arrays(next, bytes, true);
        chunks = std::move(next);
        if (!storage->pin_prefix(bytes) && storage->locked_bytes > bytes)
            throw std::runtime_error("cannot release streamed expert memory locks");
        if (bytes < capacity) madvise(static_cast<std::byte*>(storage->data) + bytes, capacity - bytes, MADV_FREE_REUSABLE);
        capacity = bytes; ++generation;
        rebuild_free();
    }
};

MlxStreamedWeightPool::MlxStreamedWeightPool(std::size_t limit, std::size_t maximum_entry)
    : impl_(std::make_unique<Impl>(limit, maximum_entry)) {}
MlxStreamedWeightPool::~MlxStreamedWeightPool() = default;
std::optional<std::size_t> MlxStreamedWeightPool::allocate(std::size_t bytes) {
    if (!bytes || bytes % impl_->page) throw std::invalid_argument("streamed expert allocation is not page aligned");
    if (bytes > impl_->limit || bytes > impl_->chunk_bytes) return std::nullopt;
    for (;;) {
        const auto found = std::find_if(impl_->free.begin(), impl_->free.end(),
            [&](const auto& range) { return range.second >= bytes; });
        if (found != impl_->free.end()) {
            const auto offset = found->first, count = found->second;
            impl_->free.erase(found);
            if (count > bytes) impl_->free.emplace(offset + bytes, count - bytes);
            impl_->occupied.emplace(offset, bytes);
            return offset;
        }
        if (impl_->capacity == impl_->limit) return std::nullopt;
        impl_->resize(std::min(impl_->limit, ((impl_->capacity / impl_->chunk_bytes) + 1) * impl_->chunk_bytes));
    }
}
void MlxStreamedWeightPool::release(std::size_t offset, std::size_t bytes) {
    const auto found = impl_->occupied.find(offset);
    if (found == impl_->occupied.end() || found->second != bytes)
        throw std::invalid_argument("streamed expert allocation was not live");
    impl_->occupied.erase(found); impl_->insert_free(offset, bytes);
}
void MlxStreamedWeightPool::set_limit(std::size_t limit) {
    limit = limit / impl_->page * impl_->page;
    if (!limit || limit > impl_->storage->bytes) throw std::invalid_argument("streamed expert budget exceeds its reserved address space");
    impl_->resize(std::min(impl_->capacity, limit)); impl_->limit = limit;
}
void MlxStreamedWeightPool::clear() {
    impl_->occupied.clear(); impl_->resize(0);
}
std::byte* MlxStreamedWeightPool::data(std::size_t offset) const {
    if (offset >= impl_->capacity) throw std::out_of_range("streamed expert offset exceeds capacity");
    return static_cast<std::byte*>(impl_->storage->data) + offset;
}
std::uint64_t MlxStreamedWeightPool::address(std::size_t offset) const {
    if (offset >= impl_->capacity) throw std::out_of_range("streamed expert offset exceeds capacity");
    return static_cast<const MTL::Buffer*>(impl_->chunks.at(offset / impl_->chunk_bytes).buffer().ptr())->gpuAddress()
        + offset % impl_->chunk_bytes;
}
std::size_t MlxStreamedWeightPool::capacity() const noexcept { return impl_->capacity; }
std::size_t MlxStreamedWeightPool::generation() const noexcept { return impl_->generation; }
std::size_t MlxStreamedWeightPool::pinned_bytes() const noexcept { return impl_->storage->locked_bytes; }
std::size_t MlxStreamedWeightPool::batch_capacity(std::size_t bytes) const noexcept {
    if (bytes < impl_->page || bytes % impl_->page) return 0;
    const auto overhead = (impl_->limit - 1) / impl_->chunk_bytes * (bytes - impl_->page);
    return overhead < impl_->limit ? (impl_->limit - overhead) / bytes : 0;
}
const std::shared_ptr<MlxWeightResidency>& MlxStreamedWeightPool::residency() const noexcept { return impl_->residency; }

} // namespace mfq::metal
