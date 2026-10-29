#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mfq {

struct MlxAffineLayout {
    std::vector<std::int64_t> shape;
    std::string scale_dtype;
    int bits = 0, group_size = 0;
    std::size_t prefix_bytes = 0;
    std::uint64_t values_bytes = 0, scales_bytes = 0;

    void validate();
    std::vector<std::byte> prefix() const;
    static MlxAffineLayout parse(const std::byte* bytes, std::size_t size);
    static std::size_t prefix_size(const std::byte* bytes, std::size_t size);
    std::uint64_t scales_offset() const { return prefix_bytes + values_bytes; }
    std::uint64_t biases_offset() const { return scales_offset() + scales_bytes; }
    std::uint64_t total_bytes() const { return biases_offset() + scales_bytes; }
};

} // namespace mfq
