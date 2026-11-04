#pragma once

#include "engine/tensor.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace mfq::weights {

enum class Layout { matrix, group, experts };
struct Shape { std::int64_t rows, cols; };
struct Placement {
    // Fixed compute devices and an additional type selected by each operation call.
    // This describes execution, not residency or accepted input devices.
    std::vector<Device> devices;
    std::optional<DeviceType> caller;
};

class Weights {
public:
    virtual ~Weights() = default;
    virtual Layout layout() const = 0;
    virtual DeviceType backend() const = 0;
    virtual std::size_t size() const = 0;
    virtual Shape shape(std::size_t index = 0) const = 0;
    // Groups use logical indices; each legal expert index reports the whole bank.
    virtual Placement placement(std::size_t index = 0) const = 0;
};

} // namespace mfq::weights
