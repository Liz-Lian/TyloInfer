#include "prefix_cache.h"

#include <stdexcept>
#include <utility>

namespace mfq::cache {

PagedPrefixCache::PagedPrefixCache(PrefixBlockStoreConfig config)
    : blocks(std::move(config)) {}

PagedPrefixCache::~PagedPrefixCache() = default;

std::size_t PagedPrefixCache::block_size_tokens() const noexcept {
    return blocks.block_size_tokens();
}

const BlockHash& PagedPrefixCache::compatibility_hash() const noexcept {
    return blocks.compatibility_hash();
}

BlockHash PagedPrefixCache::block_hash(
    const BlockHash& parent,
    const std::int64_t* token_ids,
    std::size_t token_count,
    std::string_view extra_key) const {
    return prefix_block_hash(
        compatibility_hash(), block_size_tokens(), parent, token_ids,
        token_count, extra_key);
}

PrefixMatch PagedPrefixCache::match(
    const std::vector<std::int64_t>& token_ids,
    std::string_view extra_key,
    bool record_query) {
    PrefixMatch result;
    BlockHash parent{};
    const auto block_size = block_size_tokens();
    if (record_query) ++queries_;
    for (std::size_t index = 0;
         index < token_ids.size() / block_size; ++index) {
        const auto hash = block_hash(
            parent, token_ids.data() + index * block_size,
            block_size, extra_key);
        if (!blocks.contains(hash, parent, block_size)) break;
        parent = hash;
        result.blocks.push_back(hash);
        result.matched_tokens += block_size;
    }
    for (const auto count : blocks.tail_lengths(
             parent, token_ids.size() - result.matched_tokens)) {
        const auto hash = block_hash(
            parent, token_ids.data() + result.matched_tokens,
            count, extra_key);
        if (!blocks.contains(hash, parent, count)) continue;
        result.blocks.push_back(hash);
        result.matched_tokens += count;
        break;
    }
    if (record_query && result.matched_tokens) {
        ++hits_;
        hit_tokens_ += result.matched_tokens;
    }
    return result;
}

void PagedPrefixCache::record_match(std::size_t matched_tokens) {
    ++queries_;
    if (matched_tokens) {
        ++hits_;
        hit_tokens_ += matched_tokens;
    }
}

BlockHash PagedPrefixCache::store(
    const BlockHash& parent,
    const std::int64_t* token_ids,
    std::size_t token_count,
    PagedPrefixPayload payload,
    std::string_view extra_key) {
    if (!payload) {
        throw std::invalid_argument("paged prefix payload is null");
    }
    const auto hash = block_hash(parent, token_ids, token_count, extra_key);
    return blocks.store(
        hash, parent, token_ids, token_count, std::move(payload),
        extra_key, false);
}

BlockHash PagedPrefixCache::replace(
    const BlockHash& parent,
    const std::int64_t* token_ids,
    std::size_t token_count,
    PagedPrefixPayload payload,
    std::string_view extra_key) {
    if (!payload) {
        throw std::invalid_argument("paged prefix payload is null");
    }
    const auto hash = block_hash(parent, token_ids, token_count, extra_key);
    return blocks.store(
        hash, parent, token_ids, token_count, std::move(payload),
        extra_key, true);
}

PagedPrefixCacheMetrics PagedPrefixCache::metrics() const {
    PagedPrefixCacheMetrics result;
    static_cast<PrefixBlockStoreMetrics&>(result) = blocks.metrics();
    result.queries = queries_;
    result.hits = hits_;
    result.hit_tokens = hit_tokens_;
    return result;
}

} // namespace mfq::cache
