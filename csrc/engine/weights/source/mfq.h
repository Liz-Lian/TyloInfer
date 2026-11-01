#pragma once

#include "tensor.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace mfq {

struct MfqSourceHeader {
    std::uint32_t version = 0;
    std::string architecture;
    std::unordered_map<std::string, std::string> metadata;
    std::uint32_t record_count = 0;
};

struct MfqStoredRecord {
    TensorMetadata tensor;
    std::filesystem::path source_path;
    std::uint64_t offset = 0;
    bool asset = false;
};

struct MfqFile {
    MfqSourceHeader header;
    std::vector<MfqStoredRecord> records;
};

std::filesystem::path mfq_path(const std::filesystem::path& path);
MfqFile read_mfq(const std::filesystem::path& requested);

} // namespace mfq
