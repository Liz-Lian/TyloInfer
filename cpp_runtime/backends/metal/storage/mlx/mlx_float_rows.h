#pragma once

#include <mlx/mlx.h>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace mfq::metal {

class MlxFloatRowBatch {
public:
    void append_batch(const MlxFloatRowBatch& other) {
        if (other.values_.empty()) return;
        if (this == &other || (width_ && width_ != other.width_))
            throw std::invalid_argument("floating row batch geometry differs");
        width_ = other.width_;
        values_.insert(values_.end(), other.values_.begin(), other.values_.end());
        source_bytes_ += other.source_bytes_;
    }
    mlx::core::array decode(mlx::core::Dtype dtype) const {
        if (!width_ || values_.empty() || values_.size() / width_ > INT_MAX ||
            (dtype != mlx::core::float32 && dtype != mlx::core::bfloat16 && dtype != mlx::core::float16))
            throw std::invalid_argument("invalid floating row output");
        auto value = mlx::core::array(values_.begin(),
            mlx::core::Shape{int(values_.size() / width_), width_});
        return mlx::core::astype(value, dtype);
    }
    std::size_t source_bytes_read() const noexcept { return source_bytes_; }
private:
    friend class MlxMappedFloatRows;
    int width_ = 0;
    std::size_t source_bytes_ = 0;
    std::vector<float> values_;
};

class MlxMappedFloatRows {
public:
    MlxMappedFloatRows(std::span<const std::uint8_t> bytes, std::string_view dtype) {
        if (dtype == "BF16") { kind_ = 0; element_bytes_ = 2; }
        else if (dtype == "F16") { kind_ = 1; element_bytes_ = 2; }
        else if (dtype == "F32") { kind_ = 2; element_bytes_ = 4; }
        else throw std::invalid_argument("unsupported floating row format");
        if (bytes.size() < 20) throw std::invalid_argument("truncated floating row table");
        std::uint32_t dimensions;
        std::int64_t width;
        std::memcpy(&dimensions, bytes.data(), 4);
        std::memcpy(&rows_, bytes.data() + 4, 8);
        std::memcpy(&width, bytes.data() + 12, 8);
        if (dimensions != 2 || rows_ <= 0 || width <= 0 || width > INT_MAX)
            throw std::invalid_argument("invalid floating row geometry");
        width_ = int(width);
        stride_ = std::size_t(width_) * element_bytes_;
        if (std::uint64_t(rows_) > (std::numeric_limits<std::size_t>::max() - 20) / stride_ ||
            bytes.size() != 20 + std::size_t(rows_) * stride_)
            throw std::invalid_argument("floating row payload length differs");
        values_ = bytes.data() + 20;
    }
    void append_row(std::int64_t row, MlxFloatRowBatch& batch, float scale = 1.f) const {
        if (row < 0 || row >= rows_) throw std::out_of_range("floating row is outside the table");
        if (batch.width_ && batch.width_ != width_)
            throw std::invalid_argument("floating row batch width differs");
        const auto start = batch.values_.size();
        batch.values_.resize(start + width_);
        const auto* source = values_ + std::size_t(row) * stride_;
        for (int column = 0; column < width_; ++column) {
            float value;
            if (kind_ == 0) {
                std::uint16_t bits;
                std::memcpy(&bits, source + std::size_t(column) * 2, 2);
                const auto wide = std::uint32_t(bits) << 16;
                std::memcpy(&value, &wide, 4);
            } else if (kind_ == 1) {
                mlx::core::float16_t half;
                std::memcpy(&half, source + std::size_t(column) * 2, 2);
                value = float(half);
            } else std::memcpy(&value, source + std::size_t(column) * 4, 4);
            batch.values_[start + column] = value * scale;
        }
        batch.width_ = width_;
        batch.source_bytes_ += stride_;
    }
    std::int64_t rows() const noexcept { return rows_; }
    int width() const noexcept { return width_; }
private:
    const std::uint8_t* values_ = nullptr;
    std::int64_t rows_ = 0;
    int width_ = 0, kind_ = 0, element_bytes_ = 0;
    std::size_t stride_ = 0;
};

}
