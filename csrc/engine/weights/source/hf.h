#pragma once

#include "engine/weights/source/tensor.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tyloi {

class Safetensors;

namespace hf {

struct Segment {
    std::uint64_t offset = 0;
    std::vector<std::byte> inline_bytes;
    std::string source_name;
    std::uint64_t source_offset = 0;
    std::uint64_t nbytes = 0;
};

struct Record {
    TensorMetadata metadata;
    std::vector<Segment> segments;

    void read(
        const Safetensors& files,
        std::uint64_t relative_offset,
        std::byte* destination,
        std::size_t size) const;
    std::optional<TensorFileRange> file_range(
        const Safetensors& files,
        std::uint64_t offset,
        std::size_t size) const;
};

} // namespace hf
} // namespace tyloi
