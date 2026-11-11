#pragma once

#include "requests/policy.h"
#include "state/session/snapshot.h"
#include "tensor.h"

#include <cstdint>
#include <filesystem>
#include <unordered_set>
#include <vector>

namespace tyloi::engine {

struct SessionCacheConfig {
    SessionSnapshotCacheConfig snapshots;
    bool trace = false;
};

struct PrefixCacheConfig {
    std::filesystem::path directory;
    std::uint64_t block_tokens = 256;
    std::uint64_t disk_bytes = 100ULL * 1024ULL * 1024ULL * 1024ULL;
    std::uint64_t hot_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    std::uint64_t ram_bytes = 2ULL << 30;
    std::size_t pending_writes = 64;
    std::uint64_t pending_bytes = 512ULL * 1024ULL * 1024ULL;
    bool enabled = true;
};

struct ContinuousBatchConfig {
    std::size_t max_sequences = 0;
    std::int64_t prefill_token_budget = 2048;
    bool greedy = true;
    bool cuda_graph = true;
    bool paged_kv = true;
    std::int32_t cuda_graph_minimum_tokens = 16;
};

struct GraphConfig {
    bool enabled = true;
    bool trace = false;
    std::int32_t minimum_generation_tokens = 16;
};

struct Config {
    std::vector<Device> devices;
    std::unordered_set<std::int64_t> stop_token_ids;
    std::uint64_t resident_budget_bytes = 0;
    bool online_budget = true;
    GenerationConfig generation;
    SessionCacheConfig session_cache;
    PrefixCacheConfig prefix_cache;
    bool prefill_chunk_auto = true;
    bool profile_decode = false;
    bool trace_moe = false;
    GraphConfig decode_graph;
    ContinuousBatchConfig continuous_batch;
};

std::uint64_t environment_uint64(const char* name, std::uint64_t fallback);
bool environment_enabled(const char* name, bool fallback = true);
/** Resolve runtime settings; zero prefill size selects automatic chunking. */
Config resolve_config(std::int64_t prefill_chunk_size = 0, int continuous_batching = 0);
ContinuousBatchConfig resolve_batch_config(int capacity, std::int64_t prefill_chunk_size);

} // namespace tyloi::engine
