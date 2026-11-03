#pragma once

#include <stdexcept>

namespace mfq::models {

inline void require_model(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

} // namespace mfq::models
