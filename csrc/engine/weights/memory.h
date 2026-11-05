#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace tyloi { struct TensorStorage; }
namespace tyloi::cuda { class AllocationScope; }

namespace tyloi::weights {

class Memory {
    struct Payload { std::weak_ptr<void> owner; std::size_t bytes; };
    std::vector<std::weak_ptr<tyloi::TensorStorage>> weights_;
    std::vector<Payload> ple_;

public:
    void observe_ple(const std::shared_ptr<void>& owner, std::size_t bytes);
    std::size_t resident_bytes(int device = -1) const;
    std::size_t ple_bytes() const;

private:
    friend class LoadScope;
};

class LoadScope {
    Memory* previous_;
    std::unique_ptr<tyloi::cuda::AllocationScope> allocation_;

public:
    explicit LoadScope(Memory& ledger);
    ~LoadScope();
    LoadScope(const LoadScope&) = delete;
    LoadScope& operator=(const LoadScope&) = delete;
};

void observe_streamed_ple(const std::shared_ptr<void>& owner, std::size_t bytes);
}
