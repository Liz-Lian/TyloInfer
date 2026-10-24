#pragma once

#include "backends/cpu/storage/prefix_blocks.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace mfq::cache {

struct PrefixMatch {
    std::size_t matched_tokens = 0;
    std::vector<BlockHash> blocks;
};

struct PagedPrefixCacheMetrics : PrefixBlockStoreMetrics {
    std::uint64_t queries = 0;
    std::uint64_t hits = 0;
    std::uint64_t hit_tokens = 0;
};

// Prefix identity and matching are logical state; byte ownership belongs to
// the CPU storage module.
class PagedPrefixCache {
public:
    explicit PagedPrefixCache(PrefixBlockStoreConfig config);
    ~PagedPrefixCache();

    PagedPrefixCache(const PagedPrefixCache&) = delete;
    PagedPrefixCache& operator=(const PagedPrefixCache&) = delete;

    std::size_t block_size_tokens() const noexcept;
    const BlockHash& compatibility_hash() const noexcept;
    BlockHash block_hash(
        const BlockHash& parent,
        const std::int64_t* token_ids,
        std::size_t token_count,
        std::string_view extra_key = {}) const;

    PrefixMatch match(
        const std::vector<std::int64_t>& token_ids,
        std::string_view extra_key = {},
        bool record_query = true);
    void record_match(std::size_t matched_tokens);

    BlockHash store(
        const BlockHash& parent,
        const std::int64_t* token_ids,
        std::size_t token_count,
        PagedPrefixPayload payload,
        std::string_view extra_key = {});
    BlockHash replace(
        const BlockHash& parent,
        const std::int64_t* token_ids,
        std::size_t token_count,
        PagedPrefixPayload payload,
        std::string_view extra_key = {});

    PagedPrefixCacheMetrics metrics() const;

    PrefixBlockStore blocks;

private:
    std::atomic<std::uint64_t> queries_{0};
    std::atomic<std::uint64_t> hits_{0};
    std::atomic<std::uint64_t> hit_tokens_{0};
};

} // namespace mfq::cache
