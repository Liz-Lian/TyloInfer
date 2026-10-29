#pragma once

#include <cstddef>
#include <functional>

namespace mfq::engine {
struct SessionCommand;
struct SessionResult;
}

namespace mfq::cuda {
class WeightMemoryLedger;
struct RoutedCache;
class ExpertSpace;
namespace internal {
class TextSessionCache;
}

class Budget {
public:
    Budget(WeightMemoryLedger& weight_memory, RoutedCache& weights,
        internal::TextSessionCache& cache, int primary_device);

    // The quiescent single-device owner calls this in its CUDA scope; retire runs once and is not retained.
    engine::SessionResult set(
        const engine::SessionCommand& command,
        const std::function<void()>& retire);

    std::size_t resident() const noexcept { return resident_budget_; }
    std::size_t weights() const noexcept { return weight_budget_; }
    void reclaim(std::size_t bytes, ExpertSpace* space);

private:
    WeightMemoryLedger& weight_memory_;
    RoutedCache& weights_;
    internal::TextSessionCache& cache_;
    int primary_device_;
    std::size_t resident_budget_ = 0;
    std::size_t weight_budget_ = 0;
    std::size_t initial_expert_budget_ = 0;
    std::size_t allocation_ceiling_ = 0;
};

} // namespace mfq::cuda
