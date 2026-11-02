#pragma once

#include <memory>

namespace mfq {

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
};

} // namespace mfq
