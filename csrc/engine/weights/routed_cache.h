#pragma once

#include "engine/weights/loader.h"

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

class MoeExpertCache;
class TqCache;
class TqExec;

namespace tyloi::cuda {

class WeightLoan;
class KvExpertLoan;

// One loaded model owns both cache implementations; selected banks retain their backing.
struct RoutedCache {
    RoutedCache();
    ~RoutedCache();
    RoutedCache(const RoutedCache&) = delete;
    RoutedCache& operator=(const RoutedCache&) = delete;

    std::shared_ptr<MoeExpertCache> moe;
    std::shared_ptr<TqCache> tq;
    std::optional<std::size_t> device_bytes;
    std::size_t host_bytes = 0;
    std::size_t host_limit = 0;

    tyloi::weights::Loader loader(const tyloi::source::Source& source,
        tyloi::weights::Loader::Options options, tyloi::weights::Loader::Placement placement,
        std::shared_ptr<TqExec> exec);
    std::shared_ptr<TqCache> ensure_tq(const std::shared_ptr<TqExec>& exec);
    std::vector<std::pair<std::string, double>> metrics() const;
    std::size_t budget() const;
    void resize(std::size_t bytes);
    std::size_t reserve(std::size_t bytes);
    WeightLoan borrow(std::size_t bytes);
    // Workspace operations run in the owner's CUDA scope; drain consumers before destruction.
    void workspace(int device, std::shared_ptr<KvExpertLoan> loan);
    void release_workspace(std::size_t required_available =
        std::numeric_limits<std::size_t>::max(), const char* reason = nullptr);
    void clear_workspace() noexcept;
    bool has_decode_workspace() const noexcept;
    std::size_t lend_decode(std::size_t maximum_minus_active);
    void restore_workspace(std::size_t reserve_bytes, const tyloi::source::Source* source);
    // The execution owner drains all consumers before releasing cache backing.
    void reset() noexcept;

private:
    struct Workspace;
    std::unique_ptr<Workspace> workspace_;
    std::size_t prefill_workspace_capacity() const noexcept;
    void append_workspace_metrics(std::vector<std::pair<std::string, double>>& metrics) const;
};

} // namespace tyloi::cuda
