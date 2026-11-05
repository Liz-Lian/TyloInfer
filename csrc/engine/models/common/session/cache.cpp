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

TextSessionCache::TextSessionCache(const tyloi::engine::SessionCacheConfig &session_config,
    const tyloi::engine::PrefixCacheConfig &prefix_config,
    std::shared_ptr<tyloi::cache::PagedPrefixCache> paged_cache, bool supported, int disabled_reason,
    bool native_hot_snapshots)
    : impl_(std::make_unique<Impl>(
          session_config, prefix_config, std::move(paged_cache), supported, disabled_reason,
          native_hot_snapshots)) {}

TextSessionCache::~TextSessionCache() = default;

bool TextSessionCache::persistent_prefix_enabled() const noexcept {
    return impl_->persistent_prefix_enabled();
}

void TextSessionCache::store(const std::string &session_id, TextSessionState state) {
    impl_->reserve_ram(state.bytes);
    impl_->store(session_id, std::move(state));
}

tyloi::engine::SessionResult TextSessionCache::control(const tyloi::engine::SessionCommand &command) {
    return impl_->control(command);
}

std::vector<std::pair<std::string, double>> TextSessionCache::metrics() const {
    return impl_->metrics();
}

std::map<int, std::size_t> TextSessionCache::device_hot_bytes() const {
    std::map<std::pair<int, std::uintptr_t>, std::size_t> allocations;
    const auto tensor = [&](const ::tyloi::Tensor& value) {
        for (const auto& storage : value.backings()) {
            if (!storage->device.is_cuda()) continue;
            const void* identity = storage->owner ? storage->owner.get() : storage->base;
            if (!identity) continue;
            auto& bytes = allocations[{storage->device.index, reinterpret_cast<std::uintptr_t>(identity)}];
            bytes = std::max(bytes, storage->bytes);
        }
    };
    impl_->visit_snapshots([&](const TextSessionState& state) {
        state.tensors(tensor);
    });
    std::map<int, std::size_t> result;
    for (const auto& [address, bytes] : allocations) result[address.first] += bytes;
    return result;
}

size_t TextSessionCache::clear_live_sessions() noexcept { return impl_->clear_live_sessions(); }

size_t TextSessionCache::clear() { return impl_->clear(); }

uint64_t TextSessionCache::trim_hot(uint64_t target_bytes) { return impl_->trim_hot(target_bytes); }
uint64_t TextSessionCache::set_hot_limit(uint64_t max_bytes) { return impl_->set_hot_limit(max_bytes); }
uint64_t TextSessionCache::set_ram_limit(uint64_t max_bytes) { return impl_->set_ram_limit(max_bytes); }
void TextSessionCache::reserve_ram(uint64_t bytes) { impl_->reserve_ram(bytes); }
uint64_t TextSessionCache::set_disk_limit(uint64_t max_bytes) { return impl_->set_disk_limit(max_bytes); }

} // namespace tyloi::cuda::internal
