#pragma once

#include <memory>

namespace tyloi {

class Executor {
public:
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

    virtual ~Executor() = default;
    // Destroy scopes in reverse order on the entering thread, before their resources.
    // Exiting a scope does not synchronize device work.
    virtual std::unique_ptr<Scope> enter() = 0;
    virtual std::unique_ptr<Timer> timer() = 0;
    virtual std::unique_ptr<Profile> profile(const char* name) = 0;
    // Stop concurrent submissions and call within this executor's scope.
    // Wait for executor work; owners coordinate weights and offload separately.
    virtual void drain() = 0;
};

} // namespace tyloi
