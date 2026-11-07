#include "engine/models/common/session/cache.h"
#include <cstring>

#include "engine/load.h"
#include "engine/backends/cpu/native/host_memory.h"
#include "engine/models/session_codec.h"
#include "engine/state/session/prefix_cache.h"
#include "engine/state/session/paged.h"
#include "engine/state/session/cache.h"
#include "engine/models/source.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <fstream>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#endif

namespace tyloi::cuda::internal {

bool cuda_has_separate_memory() {
    cudaDeviceProp properties{};
    TYLOINFER_CUDA_CHECK(cudaGetDeviceProperties(&properties, mfq_current_cuda_device()));
    return !properties.integrated && std::strstr(properties.name, "GB10") == nullptr;
}

std::shared_ptr<tyloi::cache::PagedPrefixCache> make_cuda_paged_prefix_cache(
    const tyloi::ModelSource &source, int64_t max_position_embeddings,
    bool supports_paged_text_session_state, const tyloi::engine::PrefixCacheConfig &config) {
    const auto format = source.metadata().find("source.format");
    // ponytail: HF source fingerprints exclude config sidecars for now.
    if (!config.enabled ||
        (format != source.metadata().end() && format->second == "hf-safetensors") ||
        !supports_paged_text_session_state) {
        return {};
    }
    tyloi::cache::PrefixBlockStoreConfig cache;
    cache.cache_dir = config.directory;
    if (!source.source_paths().empty()) cache.model_path = source.source_paths().front();
    cache.compatibility_key = cuda_prefix_cache_compatibility_key(source, max_position_embeddings);
    cache.block_size_tokens = static_cast<size_t>(config.block_tokens);
    // A longer block may lack the local/rolling state of a shorter exact
    // checkpoint. Retain those boundaries under the existing byte-budget LRU.
    cache.retain_tail_versions = 0;
    cache.max_disk_bytes = config.disk_bytes;
    const bool separate = cuda_has_separate_memory();
    cache.max_hot_bytes = separate ? config.ram_bytes : config.hot_bytes;
    cache.spill_hot_on_eviction = separate;
    if (separate) cache.available_ram_bytes = []() -> std::uint64_t {
#ifdef _WIN32
        MEMORYSTATUSEX state{};
        state.dwLength = sizeof(state);
        if (GlobalMemoryStatusEx(&state)) return state.ullAvailPhys;
#elif defined(__linux__)
        std::ifstream input("/proc/meminfo");
        std::string name, unit;
        std::uint64_t kib = 0;
        while (input >> name >> kib >> unit) if (name == "MemAvailable:") return kib * 1024;
#endif
        return UINT64_MAX;
    };
    cache.max_pending_writes = config.pending_writes;
    cache.max_pending_bytes = config.pending_bytes;
    return std::make_shared<tyloi::cache::PagedPrefixCache>(std::move(cache));
}

void CudaSessionOps::release_host_cache() {
    tyloi::trim_host_cache();
}

std::unique_ptr<tyloi::state::Sessions> make_sessions(
        const tyloi::ModelSource& source,
        const tyloi::engine::SessionCacheConfig& session_config,
        const tyloi::engine::PrefixCacheConfig& prefix_config,
        std::int64_t maximum_context, bool supported, bool paged) {
    return std::make_unique<tyloi::engine::SessionCache<CudaSessionOps>>(
        session_config, prefix_config,
        make_cuda_paged_prefix_cache(source, maximum_context, paged, prefix_config),
        supported, supported ? 0 : 1, cuda_has_separate_memory());
}

} // namespace tyloi::cuda::internal
