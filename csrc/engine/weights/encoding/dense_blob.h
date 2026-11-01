#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace mfq::dense {

struct Layout {
    std::vector<std::int64_t> shape;
    std::size_t offset = 0;
};

inline Layout parse(const std::vector<std::uint8_t>& blob, std::size_t item_size,
                    const std::optional<std::vector<std::int64_t>>& expected) {
    if (!item_size) throw std::invalid_argument("dense tensor item size must be positive");
    if (blob.size() < sizeof(std::uint32_t))
        throw std::runtime_error("truncated dense tensor header");
    std::uint32_t rank;
    std::memcpy(&rank, blob.data(), sizeof(rank));
    if (rank > (blob.size() - sizeof(rank)) / sizeof(std::int64_t))
        throw std::runtime_error("truncated dense tensor shape");
    Layout result;
    result.shape.resize(rank);
    result.offset = sizeof(rank) + rank * sizeof(std::int64_t);
    std::int64_t elements = 1;
    for (std::size_t i = 0; i < rank; ++i) {
        auto& extent = result.shape[i];
        std::memcpy(&extent, blob.data() + sizeof(rank) + i * sizeof(extent), sizeof(extent));
        if (extent < 0 || (extent && elements > std::numeric_limits<std::int64_t>::max() / extent))
            throw std::runtime_error("invalid dense tensor shape");
        elements *= extent;
    }
    if (expected && *expected != result.shape)
        throw std::runtime_error("dense tensor metadata shape mismatch");
    const auto payload = blob.size() - result.offset;
    if (payload % item_size || std::uint64_t(elements) != payload / item_size)
        throw std::runtime_error("dense tensor payload size mismatch");
    return result;
}

} // namespace mfq::dense
