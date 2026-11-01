#pragma once

#include "components/io/read_span.h"
#include "engine/weights/source/tensor.h"

#include <nlohmann/json_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

struct gguf_context;

namespace mfq {

class FileRangeReader;

bool is_gguf_file(const std::filesystem::path& path);

} // namespace mfq

namespace mfq::gguf {

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

nlohmann::json scalar(const gguf_context* ctx, std::int64_t id);
nlohmann::json value(const gguf_context* ctx, std::int64_t id);
std::vector<std::filesystem::path> paths(const std::filesystem::path& path, int splits);

} // namespace mfq::gguf
