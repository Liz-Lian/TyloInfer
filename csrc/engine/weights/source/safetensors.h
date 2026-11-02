#pragma once

#include "engine/weights/source/tensor.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mfq {

struct HfTensorLocation {
    std::size_t shard = 0;
    std::uint64_t offset = 0;
};

struct SafetensorsRecord {
    TensorMetadata metadata;
    HfTensorLocation location;
};

class Safetensors final {
public:
    explicit Safetensors(
        std::filesystem::path requested_root,
        std::string source_prefix = {});
    ~Safetensors();

    Safetensors(Safetensors&&) noexcept;
    Safetensors& operator=(Safetensors&&) noexcept;
    Safetensors(const Safetensors&) = delete;
    Safetensors& operator=(const Safetensors&) = delete;

    const std::filesystem::path& root() const noexcept;
    const std::vector<std::filesystem::path>& paths() const noexcept;
    const std::unordered_map<std::string, SafetensorsRecord>&
    records() const noexcept;
    const TensorMetadata* find(std::string_view name) const noexcept;
    void read(
        std::size_t shard,
        std::uint64_t offset,
        std::byte* destination,
        std::size_t size) const;
    void drop_file_cache() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mfq
