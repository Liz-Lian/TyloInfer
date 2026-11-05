#pragma once

#include "engine/requests/policy.h"

#include <cstddef>
#include <cstdint>
#include <memory>

class MoeExpertCache;

namespace tyloi { class ModelSource; }
namespace tyloi::engine {
struct SessionCommand;
struct SessionResult;
}

namespace tyloi::cuda {
class WeightMemoryLedger;
struct RoutedCache;
class ExpertSpace;
namespace internal {
class TextSessionCache;
}

tyloi::engine::PrefillChunk budget_batch_prefill(
        std::int64_t hidden_size, std::int64_t hc_mult, int device,
        const std::shared_ptr<MoeExpertCache>&,
        tyloi::engine::PrefillChunk, std::size_t growth = 0);

class Budget {
public:
    Budget(WeightMemoryLedger& weight_memory, RoutedCache& weights,
        internal::TextSessionCache& cache, int primary_device);

    class Change {
    public:
        Change(const Change&) = delete;
        Change& operator=(const Change&) = delete;
        engine::SessionResult apply();

    private:
        friend class Budget;
        Change(Budget&, const engine::SessionCommand&);
        Budget& budget_;
        std::size_t bytes, requested_weights, limit, dense, minimum;
        std::size_t previous_limit, previous_experts, before;
    };

    // Create to validate, then retire State and apply once in the same quiescent owner CUDA scope.
    // Budget must outlive Change.
    Change change(const engine::SessionCommand& command);

    std::size_t resident() const noexcept { return resident_budget_; }
    std::size_t weights() const noexcept { return weight_budget_; }
    // Call within the owner's execution scope.
    void reclaim(std::size_t bytes, ExpertSpace* space);
    void reserve_kv(std::size_t bytes, ExpertSpace* space);
    void reserve_ram(std::size_t bytes);
    std::int64_t prefill(std::size_t tokens, std::int64_t limit,
        std::size_t row_bytes, bool resumed, bool automatic);
    void decode(ExpertSpace* space, const ModelSource* source);

private:
    std::size_t trim_prefix(std::size_t bytes);
    void workspace(std::size_t bytes);
    WeightMemoryLedger& weight_memory_;
    RoutedCache& weights_;
    internal::TextSessionCache& cache_;
    int primary_device_;
    std::size_t resident_budget_ = 0;
    std::size_t weight_budget_ = 0;
    std::size_t initial_expert_budget_ = 0;
    std::size_t allocation_ceiling_ = 0;
};

} // namespace tyloi::cuda
