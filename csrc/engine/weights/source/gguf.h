#pragma once

#include "components/io/read_span.h"
#include "engine/weights/source/tensor.h"

#include <nlohmann/json_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tyloi {

class FileRangeReader;

bool is_gguf_file(const std::filesystem::path& path);

} // namespace tyloi

namespace tyloi::gguf {

struct ShardTensorInfo {
    std::string name;
    std::vector<std::int64_t> shape;
    int type = 0;
    std::int64_t elements = 0;
    std::uint64_t data_offset = 0;
    std::uint64_t tensor_offset = 0;
    std::uint64_t nbytes = 0;
    std::uint64_t file_size = 0;
    std::filesystem::path path;
    std::shared_ptr<FileRangeReader> file;
};

class Shard {
public:
    explicit Shard(const std::filesystem::path& path, int split_count = 1,
                   std::size_t split_no = 0);
    ~Shard();

    std::int64_t count() const;
    nlohmann::json scalar(std::string_view key) const;
    nlohmann::json metadata() const;
    std::vector<std::byte> tokenizer() const;
    std::vector<std::string> names() const;
    ShardTensorInfo tensor(std::int64_t index) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct Record {
    TensorMetadata tensor;
    std::shared_ptr<FileRangeReader> file;
    std::filesystem::path path;
    std::uint64_t offset = 0;
    std::vector<std::byte> prefix;
    std::vector<std::byte> inline_bytes;

    void read(std::uint64_t relative_offset, std::byte* destination, std::size_t size) const;
    void read(const std::vector<ReadSpan>& spans) const;
    std::optional<TensorFileRange> file_range() const;
};

int splits(const std::filesystem::path& path);
std::vector<std::filesystem::path> paths(const std::filesystem::path& path, int splits);

} // namespace tyloi::gguf
