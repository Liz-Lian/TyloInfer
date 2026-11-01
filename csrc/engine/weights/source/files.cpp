#include "engine/weights/source/files.h"

#include "components/io/async_range_reader.h"
#include "engine/weights/source/buffered.h"

#include <stdexcept>
#include <utility>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace mfq::source {

Files::Files(std::vector<std::filesystem::path> paths, FileReadMode mode)
    : paths_(std::move(paths)), mode_(mode) {
    if (mode_ == FileReadMode::Direct) {
        for (const auto& path : paths_)
            direct_.emplace(path, std::make_shared<FileRangeReader>(path, mode));
    }
}

const std::vector<std::filesystem::path>& Files::paths() const noexcept {
    return paths_;
}

FileReadMode Files::mode() const noexcept { return mode_; }

void Files::read(const std::filesystem::path& path, std::uint64_t absolute_offset,
                 std::byte* destination, std::size_t size) const {
    if (mode_ == FileReadMode::Direct)
        direct_.at(path)->read(absolute_offset, destination, size);
    else mfq::read_buffered(path, absolute_offset, destination, size);
}

std::function<void(std::uint64_t, std::byte*, std::size_t)> Files::reader(
    const std::filesystem::path& path, std::uint64_t offset,
    std::uint64_t bytes) const {
    if (mode_ == FileReadMode::Direct) {
        auto reader = direct_.at(path);
        const auto base = offset;
        return [reader, base, bytes](std::uint64_t offset,
                                     std::byte* destination, std::size_t size) {
            if (offset > bytes || size > bytes - offset)
                throw std::out_of_range("model tensor row range is out of bounds");
            reader->read(base + offset, destination, size);
        };
    }
    return mfq::buffered_reader(path, offset, bytes);
}

std::function<void(const std::vector<ReadSpan>&)> Files::batch_reader(
    const std::filesystem::path& path, std::uint64_t offset,
    std::uint64_t bytes) const {
    std::shared_ptr<AsyncRangeReader> reader;
    {
        std::lock_guard lock(mutex_);
        auto& slot = batch_[path];
        if (!slot) slot = std::make_shared<AsyncRangeReader>(path, mode_);
        reader = slot;
    }
    const auto base = offset;
    return [reader, base, bytes](const std::vector<ReadSpan>& spans) {
        std::vector<ReadSpan> absolute;
        absolute.reserve(spans.size());
        for (auto span : spans) {
            if (span.offset > bytes || span.size > bytes - span.offset)
                throw std::out_of_range("batched tensor read exceeds canonical record");
            span.offset += base;
            absolute.push_back(span);
        }
        reader->read(absolute);
    };
}

FileReadStats Files::stats() const noexcept {
    FileReadStats result;
    result.mode = mode_;
    for (const auto& [path, reader] : direct_) {
        const auto value = reader->stats();
        result.files += value.files;
        result.calls += value.calls;
        result.logical_bytes += value.logical_bytes;
        result.physical_bytes += value.physical_bytes;
        result.errors += value.errors;
        result.staging_bytes += value.staging_bytes;
        result.staging_peak_bytes += value.staging_peak_bytes;
        result.read_nanoseconds += value.read_nanoseconds;
        result.staging_cached_bytes += value.staging_cached_bytes;
        result.staging_allocations += value.staging_allocations;
    }
    {
        std::lock_guard lock(mutex_);
        for (const auto& [path, reader] : batch_) {
            const auto value = reader->stats();
            result.calls += value.calls;
            result.logical_bytes += value.logical_bytes;
            result.physical_bytes += value.physical_bytes;
            result.errors += value.errors;
            result.staging_bytes += value.staging_bytes;
            result.staging_peak_bytes += value.staging_peak_bytes;
            result.read_nanoseconds += value.read_nanoseconds;
            result.staging_cached_bytes += value.staging_cached_bytes;
            result.staging_allocations += value.staging_allocations;
        }
    }
    return result;
}

void Files::drop_cache() const noexcept {
#if !defined(_WIN32) && defined(POSIX_FADV_DONTNEED)
    for (const auto& path : paths_) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) continue;
        static_cast<void>(::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED));
        static_cast<void>(::close(fd));
    }
#endif
}

} // namespace mfq::source
