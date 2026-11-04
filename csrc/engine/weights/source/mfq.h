#pragma once

#include "tensor.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace tyloi::archive {

struct Header {
    std::uint32_t version = 0;
    std::string architecture;
    std::unordered_map<std::string, std::string> metadata;
    std::uint32_t record_count = 0;
};

struct Record {
    TensorMetadata tensor;
    std::filesystem::path source_path;
    std::uint64_t offset = 0;
    bool asset = false;
};

struct File {
    Header header;
    std::vector<Record> records;
    std::filesystem::path path;
};

inline constexpr char shard_pattern[] =
    R"(^(.*)-([0-9]{5})-of-([0-9]{5})(\.mfq|\.tq)$)";

struct Shards {
    std::filesystem::path entry;
    std::string stem;
    std::string suffix;
    std::vector<std::filesystem::path> paths;
    std::vector<bool> found;
    bool complete = false;
};

std::filesystem::path resolve(const std::filesystem::path& path);
File read(const std::filesystem::path& requested);
std::uint64_t integer(
    const Header& header,
    const std::string& key,
    std::uint64_t fallback);
Shards shards(
    const std::filesystem::path& path,
    std::uint64_t split_no,
    std::uint64_t split_count);
std::vector<std::filesystem::path> resolve(Shards parts);

struct Index {
    Header header;
    std::vector<Record> records;
    std::unordered_map<std::string, std::size_t> names;
};

class Merge {
public:
    Merge(const Header& initial, std::uint64_t split_count);

    void add(File parsed, std::uint64_t index);
    Index finish();

private:
    std::uint32_t initial_version_;
    std::string initial_architecture_;
    std::uint64_t split_count_;
    std::uint64_t expected_records_;
    std::uint64_t expected_tensors_;
    std::uint64_t actual_records_ = 0;
    std::uint64_t actual_tensors_ = 0;
    Index output_;
};

} // namespace tyloi::archive
