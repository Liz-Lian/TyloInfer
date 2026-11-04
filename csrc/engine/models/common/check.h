#pragma once

#include <stdexcept>

namespace tyloi::models {

inline void require_model(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

} // namespace tyloi::models
