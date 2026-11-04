#pragma once

#include "engine/tensor.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace tyloi::weights {

// Where a weight's compute devices live. A weight record states this itself;
// there is no shared weight base class.
struct Shape { std::int64_t rows, cols; };
struct Placement {
    // Fixed compute devices and an additional type selected by each operation call.
    // This describes execution, not residency or accepted input devices.
    std::vector<Device> devices;
    std::optional<DeviceType> caller;
};


} // namespace tyloi::weights
