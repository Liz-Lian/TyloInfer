#pragma once

#include "mlx_gguf.h"
#include <string>
#include <vector>

namespace mfq::metal {

class MlxGgufRowBatch {
public:
    void append(std::string_view dtype, int width, std::span<const std::uint8_t> bytes);
    void append_batch(const MlxGgufRowBatch& batch);
    mlx::core::array decode() const;
    std::size_t source_bytes_read() const noexcept { return bytes_.size(); }
private:
    std::string dtype_;
    int width_ = 0;
    std::size_t row_bytes_ = 0;
    std::vector<std::uint8_t> bytes_;
};

class MlxMappedGgufRows {
public:
    MlxMappedGgufRows(std::span<const std::uint8_t> bytes, std::string dtype,
        std::int64_t rows, int width);
    void append_row(std::int64_t row, MlxGgufRowBatch& batch) const;
    std::int64_t rows() const noexcept { return rows_; }
    int width() const noexcept { return width_; }
private:
    std::span<const std::uint8_t> bytes_;
    std::string dtype_;
    std::int64_t rows_;
    int width_;
    std::size_t row_bytes_;
};
} // namespace mfq::metal
