#pragma once

#include <memory>

namespace tyloi {

class Executor {
public:
    class Scope {
    public:
        virtual ~Scope() = default;
    };

    virtual ~Executor() = default;
    // Destroy scopes in reverse order on the entering thread, before their resources.
    // Exiting a scope does not synchronize device work.
    virtual std::unique_ptr<Scope> enter() = 0;
    // Stop concurrent submissions and call within this executor's scope.
    // Wait for executor work; owners coordinate weights and offload separately.
    virtual void drain() = 0;
};

} // namespace tyloi
