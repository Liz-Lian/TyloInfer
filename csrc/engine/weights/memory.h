#pragma once

#include "native/cuda/allocation.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace mfq::cuda {

class WeightMemoryLedger {
    struct Payload { std::weak_ptr<void> owner; std::size_t bytes; };
    AllocationScope::Log weights_;
    std::vector<Payload> ple_;

public:
    void observe_ple(const std::shared_ptr<void>& owner, std::size_t bytes);
    std::size_t resident_bytes(int device = -1) const;
    std::size_t ple_bytes() const;

private:
    friend class WeightMemoryLoadScope;
};

class WeightMemoryLoadScope {
    WeightMemoryLedger* previous_;
    AllocationScope allocation_;

public:
    explicit WeightMemoryLoadScope(WeightMemoryLedger& ledger);
    ~WeightMemoryLoadScope();
    WeightMemoryLoadScope(const WeightMemoryLoadScope&) = delete;
    WeightMemoryLoadScope& operator=(const WeightMemoryLoadScope&) = delete;
};

void observe_streamed_ple(const std::shared_ptr<void>& owner, std::size_t bytes);
}
