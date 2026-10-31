#pragma once
#include <cstddef>
#include <optional>

namespace mfq::cuda {
struct ProcessDeviceMemoryStats {
    std::size_t budget=0,usage=0,reservation=0;
    std::size_t shared_budget=0,shared_usage=0;
};
// Optional OS residency accounting, distinct from device-wide free VRAM.
std::optional<ProcessDeviceMemoryStats> process_device_memory_stats(int device);
}
