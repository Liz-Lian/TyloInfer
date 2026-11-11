#pragma once

#include "engine/tensor.h"
#include "engine/metrics.h"

#include <memory>
#include <span>
#include <vector>

namespace tyloi::engine { class Budget; }

namespace tyloi {

class Executor {
public:
    /** Select whether memory snapshots include diagnostic process and default-pool values. */
    enum class MemoryDetail { basic, diagnostic };
    struct Memory {
        /** Device represented by this snapshot. */
        Device device;
        /** Allocator bytes currently allocated. */
        std::size_t allocated;
        /** Allocator bytes currently reserved. */
        std::size_t reserved;
        /** Physically free device bytes. */
        std::size_t available;
        /** Total physical device bytes. */
        std::size_t total;
        /** Allocation budget in bytes; zero means no limit. */
        std::size_t limit = 0;
        /** Process-owned resident bytes, including idle allocator pages. */
        std::size_t resident = 0;
        /** Bytes available under both the allocation budget and physical memory. */
        std::size_t usable = 0;
        /** Bytes currently available in the active workspace loan. */
        std::size_t workspace = 0;
        /** Process-local device use, available only in diagnostic snapshots. */
        std::size_t local = 0;
        /** Bytes used by the device default pool in a diagnostic snapshot. */
        std::size_t default_pool_used = 0;
        /** Bytes reserved by the device default pool in a diagnostic snapshot. */
        std::size_t default_pool_reserved = 0;
    };

    class Scope {
    public:
        virtual ~Scope() = default;
    };

    // Starts on creation; stop records completion and elapsed_ms waits for it.
    class Timer {
    public:
        virtual ~Timer() = default;
        virtual void stop() = 0;
        virtual double elapsed_ms() const = 0;
    };

    // Call finish once on success; destruction restores profiling after an exception.
    class Profile {
    public:
        virtual ~Profile() = default;
        virtual void finish() = 0;
    };

    // The parent Executor outlives Graph; do not keep its Scope across suspension.
    // Invalidate before prepare, restore after each warmup, then begin and end capture.
    // end retains backend scratch; matches compares it along with the supplied owners.
    class Graph {
    public:
        virtual ~Graph() = default;
        virtual std::unique_ptr<Scope> enter() = 0;
        virtual bool matches(std::span<const Tensor> owners) const = 0;
        virtual void prepare(std::span<Tensor* const> recurrent) = 0;
        virtual void restore() = 0;
        virtual void begin() = 0;
        virtual void end(std::span<const Tensor> owners) = 0;
        virtual void replay() = 0;
        virtual void wait() = 0;
        virtual void invalidate() = 0;
    };

    virtual ~Executor() = default;
    /** Returns the device used by this executor's default execution scope. */
    virtual Device device() const noexcept = 0;
    // Destroy scopes in reverse order on the entering thread, before their resources.
    // Exiting a scope does not synchronize device work.
    virtual std::unique_ptr<Scope> enter() = 0;
    virtual std::unique_ptr<Timer> timer() = 0;
    virtual std::unique_ptr<Profile> profile(const char* name) = 0;
    virtual std::unique_ptr<Graph> graph() = 0;
    /** Query one snapshot per device in order; diagnostic details are opt-in. */
    virtual std::vector<Memory> memory(std::span<const Device> devices,
        MemoryDetail detail = MemoryDetail::basic) const = 0;
    /** Wait for the current execution stream within this executor's active scope. */
    virtual void synchronize() = 0;
    /** Return metrics owned by this executor's execution resources. */
    virtual engine::Metrics metrics() const { return {}; }
    /** Reclaim idle memory within a scope borrowing this executor's resources. */
    virtual void trim(Device device) = 0;
    /** Set the device allocation limit within its resource scope; zero removes it. */
    virtual void limit(Device device, std::size_t bytes) = 0;
    /** Bind or release the budget used by execution-owned resources. */
    virtual void bind_budget(engine::Budget*) noexcept {}
    // Stop concurrent submissions and call within this executor's scope.
    // Wait for executor work and its owned transfers; callers drain weights first.
    virtual void drain() = 0;
    // After draining and destroying graphs, clear execution caches within the scope.
    virtual void clear() = 0;
};

} // namespace tyloi
