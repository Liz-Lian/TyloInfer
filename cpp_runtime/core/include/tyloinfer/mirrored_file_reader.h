#pragma once
#include "file_range_reader.h"
#include "host_parallel.h"

namespace mfq {
// Replicas must contain the same bytes. Opening checks size and the caller's
// complete format header; it does not hash every weight byte at model load.
class MirroredFileReader final {
public:
    MirroredFileReader(std::vector<std::filesystem::path> paths,FileReadMode mode,
        std::uint64_t header_bytes=0);
    void read(std::uint64_t offset,std::byte* destination,std::size_t size) const;
    void read_coalesced(const std::vector<ReadSpan>& spans) const;
    std::vector<FileReadStats> replica_stats() const;
private:
    void validate(const ReadSpan& span) const;
    std::size_t replica(std::uint64_t offset) const noexcept;
    std::vector<std::unique_ptr<FileRangeReader>> readers_;
    std::uint64_t size_=0;
    mutable HostParallelPool pool_;
};
// Optional directories may contain only some model shards. Invalid directory
// configuration is reported; missing shards in an existing mirror are skipped.
std::vector<std::filesystem::path> model_read_replica_paths(const std::filesystem::path& primary);
}
