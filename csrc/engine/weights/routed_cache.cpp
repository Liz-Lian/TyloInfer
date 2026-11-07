#include "routed_cache.h"

#include "engine/weights/moe/expert_cache.h"
#include "engine/weights/tq/cache.h"
#include "engine/weights/workspace.h"

namespace tyloi::cuda {

tyloi::weights::Loader RoutedCache::loader(const tyloi::source::Source& source,
        tyloi::weights::Loader::Options options, tyloi::weights::Loader::Placement placement,
        std::shared_ptr<TqExec> exec) {
    return tyloi::weights::Loader(source, std::move(options), std::move(placement), *this, std::move(exec));
}

std::shared_ptr<TqCache> RoutedCache::ensure_tq(const std::shared_ptr<TqExec>& exec) {
    if (!tq) {
        const auto bytes = device_bytes.value_or(moe_expert_cache_budget_bytes(moe));
        const bool explicit_budget = device_bytes.has_value() || bool(moe);
        tq = std::make_shared<TqCache>(bytes, host_bytes, host_limit, !explicit_budget, exec);
    }
    return tq;
}

std::vector<std::pair<std::string, double>> RoutedCache::metrics() const {
    auto result = tq ? tq->metrics() : moe_expert_memory_metrics(moe);
    append_workspace_metrics(result);
    return result;
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

WeightLoan RoutedCache::borrow(std::size_t bytes) {
    return WeightLoan(moe, bytes);
}

void RoutedCache::reset() noexcept {
    clear_workspace();
    moe.reset();
    tq.reset();
    device_bytes.reset();
}

} // namespace tyloi::cuda
