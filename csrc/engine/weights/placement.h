#pragma once

#include "engine/tensor.h"
#include "engine/weights/tensor_parallel.h"

#include <cstdint>
#include <optional>
#include <vector>

struct ParallelConfig {
    std::vector<int> devices;
    std::vector<double> split;
    bool allow_duplicate_devices = false;

    bool enabled() const {
        return devices.size() > 1;
    }

    int primary_device() const {
        return devices.empty() ? 0 : devices.front();
    }
};

struct LayerPlacementConfig {
    std::vector<int> devices;
    std::vector<double> split;
    std::vector<int> layer_devices;

    bool enabled() const {
        return devices.size() > 1;
    }

    int primary_device() const {
        return devices.empty() ? 0 : devices.front();
    }

    void prepare(int64_t layers);
    int device_for_layer(int64_t layer) const;
};

namespace tyloi::weights {

/** Plan contiguous slices with a granularity supported by the configured ranks. */
inline std::vector<TensorParallelSlice> plan_slices(
    std::int64_t extent, std::int64_t preferred_granularity, const ParallelConfig& config) {
    if (!config.enabled())
        throw std::runtime_error("parallel slice planning requires at least two ranks");
    auto granularity = std::max<std::int64_t>(1, std::min<std::int64_t>(
        preferred_granularity, extent / static_cast<std::int64_t>(config.devices.size())));
    while (granularity > 1 && extent < static_cast<std::int64_t>(config.devices.size()) * granularity)
        granularity /= 2;
    auto slices = plan_tensor_parallel_slices(extent, granularity, config.devices, config.split);
    validate_tensor_parallel_slices(slices, extent, granularity);
    return slices;
}

struct Shape { std::int64_t rows, cols; };
struct Placement {
    // Fixed compute devices and an additional type selected by each operation call.
    // This describes execution, not residency or accepted input devices.
    std::vector<Device> devices;
    std::optional<DeviceType> caller;
};

inline void check_placement(const Placement& placement, Device device) {
    if (placement.caller && *placement.caller == (device.is_cuda() ? DeviceType::cuda : DeviceType::cpu))
        return;
    for (const auto& place : placement.devices)
        if (device == (place.is_cuda() ? Device(DeviceType::cuda, place.index)
                                      : Device(DeviceType::cpu))) return;
    throw std::invalid_argument("projection weight and input devices do not match");
}

} // namespace tyloi::weights
