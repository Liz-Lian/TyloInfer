#pragma once

#include <cstddef>

namespace mfq::models {
struct SnapshotSize {
    std::size_t bytes = 0;
    std::size_t peak = 0;
};
} // namespace mfq::models
