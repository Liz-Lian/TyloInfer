#pragma once

#include <cassert>
#include <cstddef>
#include <limits>
#include <list>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace mfq::weights {

// Values retain physical backing; this module owns byte admission and recency.
// A caller holding a value keeps it alive independently of cache eviction.
template<class Key, class Value, class Hash = std::hash<Key>> class Lru {
    struct Entry {
        Value value;
        std::size_t bytes;
        typename std::list<Key>::iterator recent;
    };
    std::list<Key> recent_;
    std::unordered_map<Key, Entry, Hash> entries_;
    std::optional<std::size_t> limit_;
    std::size_t used_ = 0, evictions_ = 0;

    void evict() {
        const auto entry = entries_.find(recent_.back());
        assert(entry != entries_.end() && entry->second.bytes <= used_);
        used_ -= entry->second.bytes;
        entries_.erase(entry);
        recent_.pop_back();
        ++evictions_;
    }

public:
    explicit Lru(std::optional<std::size_t> limit) : limit_(limit) {}
    Lru(const Lru&) = delete;
    Lru& operator=(const Lru&) = delete;

    std::size_t used() const noexcept { return used_; }
    std::size_t evictions() const noexcept { return evictions_; }
    std::optional<std::size_t> limit() const noexcept { return limit_; }
    bool empty() const noexcept { return entries_.empty(); }
    const auto& entries() const noexcept { return entries_; }

    Value* get(const Key& key) {
        const auto entry = entries_.find(key);
        if (entry == entries_.end()) return nullptr;
        recent_.splice(recent_.begin(), recent_, entry->second.recent);
        return &entry->second.value;
    }

    // Free cached handles before the caller allocates a new physical payload.
    bool reserve(std::size_t bytes) {
        if (limit_) {
            while (!recent_.empty() && (used_ > *limit_ || bytes > *limit_ - used_))
                evict();
            return bytes <= *limit_ - used_;
        }
        if (bytes > std::numeric_limits<std::size_t>::max() - used_)
            throw std::overflow_error("weight cache byte count overflows");
        return true;
    }

    void put(const Key& key, Value value, std::size_t bytes) {
        if (entries_.count(key)) throw std::logic_error("duplicate weight cache entry");
        if (!reserve(bytes)) return;
        recent_.push_front(key);
        try {
            entries_.emplace(key, Entry{std::move(value), bytes, recent_.begin()});
        } catch (...) {
            recent_.pop_front();
            throw;
        }
        used_ += bytes;
        assert(!limit_ || used_ <= *limit_);
    }

    std::size_t trim(std::size_t bytes) {
        const auto before = used_;
        while (!recent_.empty() && before - used_ < bytes) evict();
        return before - used_;
    }

    void limit(std::size_t bytes) {
        limit_ = bytes;
        reserve(0);
    }
};

} // namespace mfq::weights
