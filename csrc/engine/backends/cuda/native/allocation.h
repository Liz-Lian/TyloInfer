#pragma once

#include <memory>
#include <vector>

namespace mfq::cuda {
struct TensorStorage;

class AllocationScope {
public:
    using Log = std::vector<std::weak_ptr<TensorStorage>>;

    // Selecting a log does not enable recording; inner scopes enable it.
    AllocationScope() : previous_log_(log_), previous_enabled_(enabled_) {
        enabled_ = true;
    }
    explicit AllocationScope(Log& log)
        : previous_log_(log_), previous_enabled_(enabled_) {
        log_ = &log;
    }
    ~AllocationScope() {
        log_ = previous_log_;
        enabled_ = previous_enabled_;
    }

    AllocationScope(const AllocationScope&) = delete;
    AllocationScope& operator=(const AllocationScope&) = delete;

    static void record(const std::shared_ptr<TensorStorage>& storage);

private:
    static inline thread_local Log* log_ = nullptr;
    static inline thread_local bool enabled_ = false;
    Log* previous_log_;
    bool previous_enabled_;
};
}
