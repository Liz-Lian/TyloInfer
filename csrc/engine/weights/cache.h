#pragma once

#include "engine/weights/loader.h"

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::weights {
class Cache;
namespace detail {
struct CacheData;
CacheData& cache(Cache& value);
const CacheData& cache(const Cache& value);
}

// The owner drains consumers and enters the CUDA scope before workspace operations.
class Cache {
public:
    enum class Graph { native, legacy };

    Cache();
    ~Cache();
    Cache(const Cache&) = delete;
    Cache& operator=(const Cache&) = delete;

    Loader loader(const source::Source& source, Loader::Options options, Loader::Placement placement);
    std::vector<std::pair<std::string, double>> metrics() const;
    std::size_t budget() const;
    std::size_t minimum() const;
    bool graph_safe(Graph kind = Graph::native, bool parallel = false) const;
    void resize(std::size_t bytes);
    std::size_t reserve(std::size_t bytes);
    void drain() const;
    void settle();
    void trace(const char* boundary, bool wait, const source::Source* source);
    void release_workspace(std::size_t required_available =
        std::numeric_limits<std::size_t>::max(), const char* reason = nullptr);
    void clear_workspace() noexcept;
    std::size_t lend_decode(std::size_t maximum_minus_active);
    void restore_workspace(std::size_t reserve_bytes, const source::Source* source);
    void reset() noexcept;

private:
    friend detail::CacheData& detail::cache(Cache& value);
    friend const detail::CacheData& detail::cache(const Cache& value);
    std::unique_ptr<detail::CacheData> data_;
    std::size_t prefill_workspace_capacity() const noexcept;
    void append_workspace_metrics(std::vector<std::pair<std::string, double>>& metrics) const;
};
} // namespace tyloi::weights
