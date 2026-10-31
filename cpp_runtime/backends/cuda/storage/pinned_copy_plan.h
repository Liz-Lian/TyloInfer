#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace mfq::cuda {

struct PinnedUploadRange {
    const std::byte* source;
    std::byte* destination;
    std::size_t bytes;
    const void* source_allocation;
    const void* destination_allocation;
};

// CUDA rejects copies that cross distinct registrations or device allocations,
// even when their virtual addresses happen to be consecutive. Callers retain
// the allocations and identify their actual roots, rather than their slices.
inline void coalesce_pinned_uploads(std::vector<PinnedUploadRange>& ranges) {
    std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
        return reinterpret_cast<std::uintptr_t>(a.source) < reinterpret_cast<std::uintptr_t>(b.source);
    });
    std::size_t count = 0;
    for (const auto range : ranges) {
        if (!range.bytes) continue;
        if (count) {
            auto& previous = ranges[count - 1];
            const auto src = reinterpret_cast<std::uintptr_t>(previous.source);
            const auto dst = reinterpret_cast<std::uintptr_t>(previous.destination);
            if (range.source_allocation && range.destination_allocation &&
                    range.source_allocation == previous.source_allocation &&
                    range.destination_allocation == previous.destination_allocation &&
                    previous.bytes <= std::numeric_limits<std::uintptr_t>::max() - src &&
                    previous.bytes <= std::numeric_limits<std::uintptr_t>::max() - dst &&
                    src + previous.bytes == reinterpret_cast<std::uintptr_t>(range.source) &&
                    dst + previous.bytes == reinterpret_cast<std::uintptr_t>(range.destination) &&
                    range.bytes <= std::numeric_limits<std::size_t>::max() - previous.bytes) {
                previous.bytes += range.bytes;
                continue;
            }
        }
        ranges[count++] = range;
    }
    ranges.resize(count);
}

} // namespace mfq::cuda
