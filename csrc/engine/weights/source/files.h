#pragma once

#include "components/io/file_range_reader.h"
#include "components/io/read_span.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace tyloi {

class AsyncRangeReader;

namespace source {

class Files {
public:
    Files(std::vector<std::filesystem::path> paths, FileReadMode mode);

    const std::vector<std::filesystem::path>& paths() const noexcept;
    FileReadMode mode() const noexcept;
    void read(const std::filesystem::path& path, std::uint64_t absolute_offset,
              std::byte* destination, std::size_t size) const;
    std::function<void(std::uint64_t, std::byte*, std::size_t)> reader(
        const std::filesystem::path& path, std::uint64_t offset,
        std::uint64_t bytes) const;
    std::function<void(const std::vector<ReadSpan>&)> batch_reader(
        const std::filesystem::path& path, std::uint64_t offset,
        std::uint64_t bytes) const;
    FileReadStats stats() const noexcept;
    void drop_cache() const noexcept;

private:
    std::vector<std::filesystem::path> paths_;
    FileReadMode mode_;
    std::unordered_map<std::filesystem::path, std::shared_ptr<FileRangeReader>> direct_;
    mutable std::mutex mutex_;
    mutable std::unordered_map<std::filesystem::path, std::shared_ptr<AsyncRangeReader>> batch_;
};

} // namespace source
} // namespace tyloi
