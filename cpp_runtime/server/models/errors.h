#pragma once

#include <stdexcept>
#include <string>

namespace mfq::server {
struct ModelError : std::runtime_error {
    enum class Kind { invalid, not_found, conflict, unavailable, stopping, cancelled, too_large, failure };
    Kind kind;
    const char* code;
    ModelError(Kind kind, const char* code, const std::string& message)
        : std::runtime_error(message), kind(kind), code(code) {}
};
}
