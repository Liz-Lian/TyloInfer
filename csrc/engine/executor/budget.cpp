#include "engine/executor/budget.h"

#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/models/common/session/cache.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/weights/routed_cache.h"
#include "engine/executor/engine.h"
#include "engine/weights/workspace.h"
#include "engine/weights/memory.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace mfq::cuda {

Budget::Budget(WeightMemoryLedger& weight_memory, RoutedCache& weights,
        internal::TextSessionCache& cache, int primary_device)
    : weight_memory_(weight_memory), weights_(weights), cache_(cache),
      primary_device_(primary_device), initial_expert_budget_(weights.budget()),
      allocation_ceiling_(default_context(primary_device)->memory_stats().limit) {}

void Budget::reclaim(std::size_t bytes, ExpertSpace* space) {
    auto context=default_context(primary_device_);
    const auto available=context->available_memory();
    if(available>=bytes)return;
    if(space && !space->empty()) {
        std::cerr<<"runtime_workspace_cache_reclaim reason=workspace requested_bytes="<<bytes
            <<" available_bytes="<<available<<'\n';
        space->release(bytes);
    }
    if(context->available_memory()>=bytes)return;
    std::size_t hot=0;
    for(const auto& [name,value]:cache_.metrics())
        if(name=="prefix_cache_hot_bytes")hot=std::size_t(value);
    const auto shortage=bytes-context->available_memory();
    const auto released=cache_.trim_hot(hot>shortage?hot-shortage:0);
    context->reclaim_idle_memory();
    if(released)std::cerr<<"runtime_workspace_reclaim requested_bytes="<<bytes
        <<" available_bytes="<<context->available_memory()<<" prefix_released_bytes="<<released<<'\n';
}

engine::SessionResult Budget::set(
        const engine::SessionCommand& command,
        const std::function<void()>& retire) {
    const auto context = default_context(primary_device_);
    const auto dense = weight_memory_.resident_bytes();
    const auto minimum = moe_expert_cache_minimum_bytes(weights_.moe);
    const auto requested_weights = command.weight_bytes.value_or(weight_budget_);
    if (requested_weights && requested_weights < dense + minimum)
        throw std::invalid_argument("weight budget is below dense weights and minimum expert working set");
    auto limit = command.bytes ? allocation_ceiling_
            ? std::min<std::size_t>(command.bytes, allocation_ceiling_)
            : command.bytes
        : allocation_ceiling_;
    if (limit) {
        std::size_t free = 0, total = 0;
        TYLOINFER_CUDA_CHECK(cudaMemGetInfo(&free, &total));
        limit = std::min(limit, std::min(total, free + context->resident_memory_usage()));
    }
    if (limit && limit < dense + minimum)
        throw std::invalid_argument("resident memory budget is below non-offloadable weights");
    const auto previous_limit = context->memory_stats().limit;
    const auto previous_experts = weights_.budget();
    const auto before = context->resident_memory_usage();

    retire();
    release_moe_decode_buffers(weights_.moe);
    context->reclaim_idle_memory();
    auto target = requested_weights ? requested_weights - dense : initial_expert_budget_;
    std::size_t hot = 0, expert_bytes = 0;
    for (const auto& [name, value] : cache_.metrics())
        if (name == "prefix_cache_hot_bytes") hot = std::size_t(value);
    for (const auto& [name, value] : weights_.metrics())
        if (name == "resident_expert_cache_bytes") expert_bytes = std::size_t(value);
    auto used = context->resident_memory_usage();
    if (limit && used > limit) {
        cache_.trim_hot(hot > used - limit ? hot - (used - limit) : 0);
        context->reclaim_idle_memory();
        used = context->resident_memory_usage();
    }
    const auto fixed = used > expert_bytes ? used - expert_bytes : 0;
    if (limit) {
        if (fixed > limit || minimum > limit - fixed)
            throw std::invalid_argument("resident memory budget is below live KV and workspace requirements");
        target = std::min(target, limit - fixed);
    }
    try {
        if (!limit || limit > previous_limit) context->set_memory_limit(limit);
        weights_.resize(std::max(target, minimum));
        context->set_memory_limit(limit);
    } catch (...) {
        context->set_memory_limit(0);
        weights_.resize(previous_experts);
        context->set_memory_limit(previous_limit);
        throw;
    }
    resident_budget_ = command.bytes;
    weight_budget_ = requested_weights;
    const auto after = context->resident_memory_usage();
    return {before > after ? before - after : 0, {}};
}

} // namespace mfq::cuda
