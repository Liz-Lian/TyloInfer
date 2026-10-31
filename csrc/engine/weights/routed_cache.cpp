#include "routed_cache.h"

#include "engine/weights/moe/expert_cache.h"
#include "engine/weights/tq/cache.h"

namespace mfq::cuda {

std::shared_ptr<TqCache> RoutedCache::ensure_tq(const std::shared_ptr<TqExec>& exec) {
    if (!tq) {
        const auto bytes = device_bytes.value_or(moe_expert_cache_budget_bytes(moe));
        const bool explicit_budget = device_bytes.has_value() || bool(moe);
        tq = std::make_shared<TqCache>(bytes, host_bytes, host_limit, !explicit_budget, exec);
    }
    return tq;
}

std::vector<std::pair<std::string, double>> RoutedCache::metrics() const {
    return tq ? tq->metrics() : moe_expert_memory_metrics(moe);
}

std::size_t RoutedCache::budget() const {
    return tq ? tq->budget() : moe_expert_cache_budget_bytes(moe);
}

void RoutedCache::resize(std::size_t bytes) {
    if (tq) tq->resize(bytes);
    else resize_moe_expert_cache(moe, bytes);
}

std::size_t RoutedCache::reserve(std::size_t bytes) {
    return tq ? tq->reserve(bytes) : reclaim_moe_optional_host(moe, bytes);
}

void RoutedCache::reset() noexcept {
    moe.reset();
    tq.reset();
    device_bytes.reset();
}

} // namespace mfq::cuda
