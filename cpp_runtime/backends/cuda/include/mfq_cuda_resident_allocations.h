#pragma once
#include <cstddef>
#include <optional>

namespace mfq::cuda {
// Remaining WDDM shared-memory budget; absent on platforms without WDDM.
std::optional<std::size_t> shared_memory_available(int device);
// Keep long-lived request state out of the temporary async allocator pool.
class ResidentAllocationScope {
public:
    explicit ResidentAllocationScope(bool enabled=true) noexcept;
    ~ResidentAllocationScope() noexcept { finish(); }
    ResidentAllocationScope(const ResidentAllocationScope&)=delete;
    ResidentAllocationScope& operator=(const ResidentAllocationScope&)=delete;
    void finish() noexcept;
private:
    bool previous_,active_=true;
};
// Evictable routed images must not share allocator pages with model weights.
class RoutedWeightAllocationScope {
public:
    explicit RoutedWeightAllocationScope(bool enabled=true) noexcept;
    ~RoutedWeightAllocationScope() noexcept;
    RoutedWeightAllocationScope(const RoutedWeightAllocationScope&)=delete;
    RoutedWeightAllocationScope& operator=(const RoutedWeightAllocationScope&)=delete;
private:
    ResidentAllocationScope resident_;
    bool previous_;
};
}
