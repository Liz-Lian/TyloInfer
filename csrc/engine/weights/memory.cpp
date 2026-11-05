#include "engine/weights/memory.h"

#include "engine/backends/cuda/native/allocation.h"
#include "engine/tensor.h"

namespace tyloi::weights {
namespace {
thread_local Memory* loading_weight_memory = nullptr;
}

void Memory::observe_ple(const std::shared_ptr<void>& owner, std::size_t bytes) {
    ple_.push_back({owner, bytes});
}

std::size_t Memory::resident_bytes(int device) const {
    std::size_t bytes = 0;
    for (const auto& weak : weights_)
        if (auto storage = weak.lock())
            if (device < 0 || storage->device.index == device)
                bytes += storage->bytes;
    return bytes;
}

std::size_t Memory::ple_bytes() const {
    std::size_t bytes = 0;
    for (const auto& payload : ple_)
        if (!payload.owner.expired())
            bytes += payload.bytes;
    return bytes;
}

LoadScope::LoadScope(Memory& ledger)
    : previous_(loading_weight_memory), allocation_(std::make_unique<tyloi::cuda::AllocationScope>(ledger.weights_)) {
    loading_weight_memory = &ledger;
}

LoadScope::~LoadScope() {
    loading_weight_memory = previous_;
}

void observe_streamed_ple(const std::shared_ptr<void>& owner, std::size_t bytes) {
    if (loading_weight_memory)
        loading_weight_memory->observe_ple(owner, bytes);
}
}
