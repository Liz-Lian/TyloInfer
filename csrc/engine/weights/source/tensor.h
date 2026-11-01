#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfq {

struct TensorMetadata {
    std::string name;
    std::string dtype;
    std::string stored_dtype;
    std::optional<std::vector<std::int64_t>> shape;
    std::uint64_t nbytes = 0;
};

struct TensorFileRange {
    std::filesystem::path path;
    std::uint64_t offset = 0;
    std::uint64_t nbytes = 0;
};

// A capability refusal, distinct from invalid metadata or a failed read.
class TensorReaderUnsupported : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

} // namespace mfq
