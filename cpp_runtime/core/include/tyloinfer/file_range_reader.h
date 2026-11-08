#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>
#include "read_span.h"

namespace mfq {

enum class FileReadMode { Buffered, Direct };
FileReadMode model_file_read_mode();
const char* file_read_mode_name(FileReadMode mode) noexcept;

struct FileReadStats {
    FileReadMode mode = FileReadMode::Buffered;
    std::uint64_t files = 0;
    std::uint64_t calls = 0;
    std::uint64_t logical_bytes = 0;
    std::uint64_t physical_bytes = 0;
    std::uint64_t errors = 0;
    std::uint64_t staging_bytes = 0;
    // Summing this across files is a conservative bound, not a simultaneous peak.
    std::uint64_t staging_peak_bytes = 0;
    std::uint64_t read_nanoseconds = 0;
    std::uint64_t staging_cached_bytes = 0;
    std::uint64_t staging_allocations = 0;
};

inline void add_file_read_stats(FileReadStats& result,const FileReadStats& value) noexcept {
    if(value.files)result.mode=value.mode;
    result.files+=value.files;result.calls+=value.calls;
    result.logical_bytes+=value.logical_bytes;result.physical_bytes+=value.physical_bytes;
    result.errors+=value.errors;result.staging_bytes+=value.staging_bytes;
    result.staging_peak_bytes+=value.staging_peak_bytes;result.read_nanoseconds+=value.read_nanoseconds;
    result.staging_cached_bytes+=value.staging_cached_bytes;result.staging_allocations+=value.staging_allocations;
}

// Owning positional reads. Direct mode never silently reverts to buffered I/O.
// Arbitrary byte ranges use aligned temporary storage. Readers share a bounded
// idle pool with at most 8 MiB per arena. Only storage is reused: every
// request reads the file, without a shared file-position lock.
// Linux batches use a per-worker native AIO queue when supported by the file
// system; otherwise the same Direct descriptor is read on that SSD worker.
class FileRangeReader;
struct FileReadRange {
    std::shared_ptr<const FileRangeReader> reader;
    ReadSpan span;
};

class FileRangeReader final {
public:
    FileRangeReader(const std::filesystem::path& path, FileReadMode mode);
    ~FileRangeReader();
    FileRangeReader(const FileRangeReader&) = delete;
    FileRangeReader& operator=(const FileRangeReader&) = delete;
    void read(std::uint64_t offset, std::byte* destination, std::size_t size) const;
    void read_contiguous(const std::vector<ReadSpan>& spans) const;
    // Submit independent ranges together; all requests finish before returning.
    void read_batch(const std::vector<ReadSpan>& spans) const;
    // Coalesce overlapping/adjacent source ranges, then scatter in caller order.
    // The aligned read buffer is shared with direct I/O, avoiding a second copy.
    void read_coalesced(const std::vector<ReadSpan>& spans) const;
    // Ranges from independent retained files share one bounded I/O queue.
    static void read_many(const std::vector<FileReadRange>& ranges);
    std::uint64_t size() const noexcept;
    FileReadStats stats() const noexcept;
    std::size_t alignment() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mfq
