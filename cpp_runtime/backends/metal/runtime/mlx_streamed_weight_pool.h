#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace mfq::metal {
class MlxWeightResidency;

class MlxStreamedWeightPool {
public:
    MlxStreamedWeightPool(std::size_t limit, std::size_t maximum_entry);
    ~MlxStreamedWeightPool();
    MlxStreamedWeightPool(const MlxStreamedWeightPool&) = delete;
    MlxStreamedWeightPool& operator=(const MlxStreamedWeightPool&) = delete;
    std::optional<std::size_t> allocate(std::size_t bytes);
    void release(std::size_t offset, std::size_t bytes);
    void set_limit(std::size_t limit);
    void clear();
    std::byte* data(std::size_t offset) const;
    std::uint64_t address(std::size_t offset) const;
    std::size_t capacity() const noexcept;
    std::size_t generation() const noexcept;
    std::size_t pinned_bytes() const noexcept;
    std::size_t batch_capacity(std::size_t entry_bytes) const noexcept;
    const std::shared_ptr<MlxWeightResidency>& residency() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mfq::metal
