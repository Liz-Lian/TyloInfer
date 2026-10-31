#include "engine/executor/load.h"

#include "engine/state/kv_memory.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/weights/routed_cache.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/backends/cuda/native/tensor/backend.h"

#include <utility>

namespace mfq::cuda {

void finalize_kv(KvMemoryManager* kv, RoutedCache& weights,
        std::shared_ptr<KvOffloadStore>& offload) {
    if (kv) {
        kv->finalize(moe_expert_cache_minimum_bytes(weights.moe));
        set_moe_kv_expert_loan(weights.moe, kv->expert_loan());
        if (auto store = kv->offload()) offload = std::move(store);
    }
}

std::chrono::steady_clock::time_point finish_load(
        KvMemoryManager* kv, RoutedCache& weights, std::size_t graph_reserve,
        bool report_memory) {
    if (moe_expert_cache_has_sources(weights.moe) &&
            !moe_expert_cache_finalized(weights.moe)) {
        finalize_moe_expert_cache(weights.moe,
            (kv ? kv->growth_reserve() : 0) + graph_reserve,
            kv ? kv->expert_budget_charge() : 0);
    }
    mfq_cuda_synchronize();
    const auto loaded = std::chrono::steady_clock::now();
    report_cuda_memory(report_memory, "loaded");
    return loaded;
}

} // namespace mfq::cuda
