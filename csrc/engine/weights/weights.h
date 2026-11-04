#pragma once

#include "engine/tensor.h"

#include <cstddef>
#include <cstdint>

namespace mfq::weights {

enum class Layout { matrix, group, experts };
struct Shape { std::int64_t rows, cols; };

class Weights {
public:
    virtual ~Weights() = default;
    virtual Layout layout() const = 0;
    virtual DeviceType backend() const = 0;
    virtual std::size_t size() const = 0;
    virtual Shape shape(std::size_t index = 0) const = 0;
};

} // namespace mfq::weights
