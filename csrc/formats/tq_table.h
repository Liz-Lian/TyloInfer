#pragma once

#include "formats/tq_rows.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace mfq {

std::vector<std::uint8_t> join_rows(
    const std::vector<std::vector<std::uint8_t>>& blobs);

// A logical row table assembled from independently readable TQ segments.
// It owns only segment readers and contains no model-source or device state.
class TqTable {
    std::vector<std::shared_ptr<TqRows>> segments_;
    std::vector<std::uint64_t> ends_;
    std::int64_t columns_ = 0;
    std::uint64_t storage_bytes_ = 0;

public:
    TqTable(std::vector<std::shared_ptr<TqRows>> segments,
            std::int64_t columns, std::uint64_t expected_rows,
            std::uint64_t storage_bytes);

    std::uint64_t rows() const noexcept { return ends_.back(); }
    std::int64_t width() const noexcept { return columns_; }
    std::uint64_t storage_bytes() const noexcept { return storage_bytes_; }
    std::size_t row_image_bytes_bound() const;
    std::vector<std::uint8_t> slice_rows(std::uint64_t begin,
                                         std::uint64_t end) const;

    std::size_t segment_count() const noexcept { return segments_.size(); }
    std::uint64_t segment_end(std::size_t index) const { return ends_.at(index); }
    const TqRows& segment(std::size_t index) const { return *segments_.at(index); }
};

} // namespace mfq
