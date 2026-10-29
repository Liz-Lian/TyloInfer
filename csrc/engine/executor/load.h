#pragma once

#include <chrono>
#include <cstddef>
#include <memory>

namespace mfq::cuda {

class KvMemoryManager;
class KvOffloadStore;
struct RoutedCache;

void finalize_kv(KvMemoryManager* kv, RoutedCache& weights,
    std::shared_ptr<KvOffloadStore>& offload);
std::chrono::steady_clock::time_point finish_load(
    KvMemoryManager* kv, RoutedCache& weights, std::size_t graph_reserve,
    bool report_memory);

} // namespace mfq::cuda
