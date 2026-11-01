#pragma once

#include "components/io/read_span.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <vector>

namespace mfq {

// File hints follow the operation, not the reader's lifetime. Bulk prefill
// opts in on each read worker; latency-sensitive demand reads use memcpy.
class MappedPrefetchScope {
public:
    explicit MappedPrefetchScope(bool enabled, bool release_after_read=false);
    ~MappedPrefetchScope();
    MappedPrefetchScope(const MappedPrefetchScope&)=delete;
    MappedPrefetchScope& operator=(const MappedPrefetchScope&)=delete;
private:
    bool previous_;
    bool previous_release_;
};

// Process-local totals enabled by TYLOINFER_PROFILE_SSD_EXPERTS: prefetch ns,
// memcpy ns, bytes copied, range reads, and batch prefetch calls.
std::array<std::uint64_t,5> mapped_tensor_read_profile();

// Advise the OS to reclaim clean mapped source pages after weights have
// been copied into their resident tier. Readers retain their mappings and
// can fault pages back in. Returns the number of live views advised.
std::size_t release_mapped_tensor_pages();

std::function<void(std::uint64_t, std::byte*, std::size_t)> mapped_reader(const std::filesystem::path& path, std::uint64_t offset, std::uint64_t bytes);
std::function<void(const std::vector<ReadSpan>&)> mapped_batch_reader(const std::filesystem::path& path, std::uint64_t offset, std::uint64_t bytes);

} // namespace mfq
