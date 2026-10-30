#include "engine/weights/memory.h"

#include "engine/backends/cuda/native/tensor/tensor.h"

namespace mfq::cuda {
namespace {
thread_local WeightMemoryLedger* loading_weight_memory = nullptr;
}

void WeightMemoryLedger::observe_ple(const std::shared_ptr<void>& owner, std::size_t bytes) {
    ple_.push_back({owner, bytes});
}

std::size_t WeightMemoryLedger::resident_bytes(int device) const {
    std::size_t bytes = 0;
    for (const auto& weak : weights_)
        if (auto storage = weak.lock())
            if (device < 0 || storage->device.index == device)
                bytes += storage->bytes;
    return bytes;
}

std::size_t WeightMemoryLedger::ple_bytes() const {
    std::size_t bytes = 0;
    for (const auto& payload : ple_)
        if (!payload.owner.expired())
            bytes += payload.bytes;
    return bytes;
}

WeightMemoryLoadScope::WeightMemoryLoadScope(WeightMemoryLedger& ledger)
    : previous_(loading_weight_memory), allocation_(ledger.weights_) {
    loading_weight_memory = &ledger;
}

WeightMemoryLoadScope::~WeightMemoryLoadScope() {
    loading_weight_memory = previous_;
}

void observe_streamed_ple(const std::shared_ptr<void>& owner, std::size_t bytes) {
    if (loading_weight_memory)
        loading_weight_memory->observe_ple(owner, bytes);
}
}
