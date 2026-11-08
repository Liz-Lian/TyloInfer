#pragma once

#include "engine/requests/policy.h"
#include "engine/tensor.h"
#include "engine/weights/cache.h"

#include <cstddef>
#include <cstdint>

namespace tyloi::state { class Sessions; }
namespace tyloi::weights { class Memory; }
namespace tyloi::engine {
struct SessionCommand;
struct SessionResult;

class Budget {
public:
    Budget(::tyloi::weights::Memory& weight_memory, ::tyloi::weights::Cache& weights,
        state::Sessions& cache, Device primary);
    ~Budget();
    Budget(const Budget&) = delete;
    Budget& operator=(const Budget&) = delete;

    // Referenced owners outlive Budget; bind within their execution scope and unbind before retirement.
    void bind(bool predictor);
    void unbind() noexcept;

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
    void reclaim(std::size_t bytes);
    void reclaim_kv(Device device, std::size_t bytes);
    void reserve_kv(Device device, std::size_t bytes);
    void reserve_ram(std::size_t bytes);
    void workspace(Device device, std::size_t bytes);
    PrefillChunk batch(Device device, std::size_t row_bytes, PrefillChunk chunk,
        std::size_t growth = 0);
    void request(Device device, bool retire_decode);
    void prepare_prefill(Device device, std::size_t tokens, std::int64_t limit,
        ::tyloi::weights::Cache::Prefill mode, bool retire_decode = false);
    std::int64_t prefill(std::size_t tokens, std::int64_t limit,
        std::size_t row_bytes, bool resumed, bool automatic);
    std::int64_t chunk(Device device, std::size_t tokens, std::int64_t limit,
        std::size_t row_bytes, bool resumed, bool automatic, bool allow_borrow = false);
    bool snapshot(Device device, std::size_t bytes, std::size_t peak);

private:
    std::size_t trim_prefix(std::size_t bytes, int device);
    ::tyloi::weights::Memory& weight_memory_;
    ::tyloi::weights::Cache& weights_;
    state::Sessions& cache_;
    int primary_device_;
    std::size_t resident_budget_ = 0;
    std::size_t weight_budget_ = 0;
    std::size_t initial_expert_budget_ = 0;
    std::size_t allocation_ceiling_ = 0;
};

} // namespace tyloi::engine
