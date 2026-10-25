#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace mfq::mfe {

struct Part {
    std::size_t offset = 0;
    std::size_t bytes = 0;
};

struct Entry {
    std::vector<std::int32_t> ids;
    std::string dtype;
    Part runtime;
    Part payload;
};

struct Index {
    int version = 0;
    int experts = 0;
    int output = 0;
    int width = 0;
    std::vector<Entry> pools;
};

// Reads metadata only; payload offsets refer to the caller's immutable source.
Index parse(std::size_t bytes,
            const std::function<void(std::size_t, std::uint8_t*, std::size_t)>& read);

} // namespace mfq::mfe
