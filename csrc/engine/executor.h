#pragma once

#include "engine/tensor.h"

#include <memory>
#include <span>
#include <vector>

namespace tyloi {

class Executor {
public:
    struct Memory {
        Device device;
        std::size_t allocated, reserved, available, total;
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
        virtual bool matches(std::span<const std::shared_ptr<const Tensor>> owners) const = 0;
        virtual void prepare(std::span<Tensor* const> recurrent) = 0;
        virtual void restore() = 0;
        virtual void begin() = 0;
        virtual void end(std::span<const std::shared_ptr<const Tensor>> owners) = 0;
        virtual void replay() = 0;
        virtual void wait() = 0;
        virtual void invalidate() = 0;
    };

    virtual ~Executor() = default;
    // Destroy scopes in reverse order on the entering thread, before their resources.
    // Exiting a scope does not synchronize device work.
    virtual std::unique_ptr<Scope> enter() = 0;
    virtual std::unique_ptr<Timer> timer() = 0;
    virtual std::unique_ptr<Profile> profile(const char* name) = 0;
    virtual std::unique_ptr<Graph> graph() = 0;
    // Query allocator and physical byte snapshots for the requested devices within this executor's scope.
    virtual std::vector<Memory> memory(std::span<const Device> devices) const = 0;
    // Stop concurrent submissions and call within this executor's scope.
    // Wait for executor work; owners coordinate weights and offload separately.
    virtual void drain() = 0;
    // After draining and destroying graphs, clear execution caches within the scope.
    virtual void clear() = 0;
};

} // namespace tyloi
