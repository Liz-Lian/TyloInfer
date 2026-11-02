#include "hf.h"

#include "safetensors.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace mfq::hf {

void Record::read(
    const Safetensors& files,
    std::uint64_t relative_offset,
    std::byte* destination,
    std::size_t size) const {
    if (relative_offset > metadata.nbytes ||
        size > metadata.nbytes - relative_offset) {
        throw std::out_of_range("model tensor byte range is out of bounds: " +
                                metadata.name);
    }
    const auto request_end = relative_offset + size;
    for (const auto& segment : segments) {
        const auto segment_size = segment.inline_bytes.empty()
            ? segment.nbytes : segment.inline_bytes.size();
        if (segment_size > std::numeric_limits<std::uint64_t>::max() -
                           segment.offset) {
            throw std::overflow_error(
                "HF virtual tensor segment byte size overflow");
        }
        const auto segment_end = segment.offset + segment_size;
        const auto begin = std::max(relative_offset, segment.offset);
        const auto end = std::min<std::uint64_t>(request_end, segment_end);
        if (begin >= end) continue;
        const auto target_offset = static_cast<std::size_t>(begin - relative_offset);
        const auto count = static_cast<std::size_t>(end - begin);
        if (!segment.inline_bytes.empty()) {
            std::copy_n(
                segment.inline_bytes.data() + (begin - segment.offset), count,
                destination + target_offset);
        } else {
            const auto found = files.records().find(segment.source_name);
            if (found == files.records().end()) {
                throw std::runtime_error(
                    "model tensor not found: " + segment.source_name);
            }
            const auto& source = found->second;
            const auto source_offset =
                segment.source_offset + begin - segment.offset;
            if (source_offset > source.metadata.nbytes ||
                count > source.metadata.nbytes - source_offset) {
                throw std::out_of_range(
                    "model tensor byte range is out of bounds: " +
                    segment.source_name);
            }
            files.read(
                source.location.shard,
                source.location.offset + source_offset,
                destination + target_offset, count);
        }
    }
}

std::optional<TensorFileRange> Record::file_range(
    const Safetensors& files,
    std::uint64_t offset,
    std::size_t size) const {
    if (offset > metadata.nbytes || size > metadata.nbytes - offset) {
        throw std::out_of_range("HF tensor file range is out of bounds");
    }
    for (const auto& segment : segments) {
        if (!segment.inline_bytes.empty() || offset < segment.offset ||
            offset - segment.offset > segment.nbytes ||
            size > segment.nbytes - (offset - segment.offset)) continue;
        const auto found = files.records().find(segment.source_name);
        if (found == files.records().end()) {
            throw std::runtime_error(
                "model tensor not found: " + segment.source_name);
        }
        const auto& location = found->second.location;
        return TensorFileRange{
            files.paths().at(location.shard),
            location.offset + segment.source_offset + offset - segment.offset,
            size};
    }
    return std::nullopt;
}

} // namespace mfq::hf
