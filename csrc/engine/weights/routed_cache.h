#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

class MoeExpertCache;
class TqCache;
class TqExec;

namespace mfq::cuda {

// One loaded model owns both cache implementations; selected banks retain their backing.
struct RoutedCache {
    RoutedCache() = default;
    RoutedCache(const RoutedCache&) = delete;
    RoutedCache& operator=(const RoutedCache&) = delete;

    std::shared_ptr<MoeExpertCache> moe;
    std::shared_ptr<TqCache> tq;
    std::optional<std::size_t> device_bytes;
    std::size_t host_bytes = 0;
    std::size_t host_limit = 0;

    std::shared_ptr<TqCache> ensure_tq(const std::shared_ptr<TqExec>& exec);
    std::vector<std::pair<std::string, double>> metrics() const;
    std::size_t budget() const;
    void resize(std::size_t bytes);
    std::size_t reserve(std::size_t bytes);
    // The execution owner drains all consumers before releasing cache backing.
    void reset() noexcept;
};

} // namespace mfq::cuda
