#pragma once

#include "engine/tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace tyloi::encoding {

struct Tq {
    std::size_t words = 0, words_bytes = 0;
    std::size_t scales = 0, scales_bytes = 0;
    std::size_t relative = 0, bytes = 0;
    std::int64_t stored_columns = 0;
    int gslog = 0;
    bool columns = false;
};

// TQ-encoded representation: logical [rows, columns] F16 over a packed uint8
// payload. The format fields and their validation live here; the backing
// buffer and its identity stay in the shared TensorStorage.
class TqTensor final : public TensorImpl {
public:
    TqTensor() = default;

    TqTensor(const DenseTensor& image, std::int64_t rows, std::int64_t columns, Tq layout);

    const Tq& layout() const noexcept { return layout_; }
    // Dense uint8 view of the packed payload sharing the backing storage.
    const DenseTensor& image() const noexcept { return image_; }

    bool defined() const noexcept override { return storage_ != nullptr; }
    std::int64_t dim() const noexcept override { return storage_ ? 2 : 0; }
    std::int64_t numel() const override { return storage_ ? rows_ * columns_ : 0; }
    std::size_t nbytes() const override {
        return storage_ ? static_cast<std::size_t>(rows_ * columns_) * 2 : 0;
    }
    std::size_t backing_nbytes() const override { return storage_ ? storage_->bytes : 0; }
    std::int64_t size(std::int64_t dimension) const override {
        if (dimension == 0) return rows_;
        if (dimension == 1) return columns_;
        throw std::out_of_range("tensor dimension is out of range");
    }
    ScalarType scalar_type() const noexcept override { return kFloat16; }
    const Device& device() const noexcept override { return device_; }
    bool is_contiguous() const noexcept override { return false; }
    bool is_encoded() const noexcept override { return true; }
    std::vector<const DenseTensor*> buffers() const override { return {&image_}; }
    const std::shared_ptr<TensorStorage>& storage() const noexcept { return storage_; }
    std::size_t byte_offset() const noexcept { return byte_offset_; }
    void record_stream(std::uintptr_t stream) const override {
        if (storage_ && storage_->record_stream) storage_->record_stream(stream);
    }
    std::shared_ptr<TensorImpl> relocated(Device target) const override {
        auto moved = std::make_shared<TqTensor>(*this);
        if (image_.device() != target) {
            moved->image_ = image_.to(target);
            moved->storage_ = moved->image_.storage();
            moved->device_ = moved->image_.device();
            moved->byte_offset_ = moved->image_.byte_offset();
        }
        return moved;
    }

private:
    DenseTensor image_;
    std::shared_ptr<TensorStorage> storage_;
    Device device_{};
    std::size_t byte_offset_ = 0;
    std::int64_t rows_ = 0, columns_ = 0;
    Tq layout_{};
};

inline TqTensor::TqTensor(
    const DenseTensor& image, std::int64_t rows, std::int64_t columns, Tq layout) {
    if (!image.defined() || image.is_encoded() || image.scalar_type() != kUInt8 ||
        image.dim() != 1 || !image.is_contiguous() || rows < 1 || columns < 1 ||
        layout.stored_columns < columns || layout.stored_columns > INT32_MAX || rows > INT32_MAX ||
        layout.bytes > static_cast<std::size_t>(image.numel()) ||
        image.byte_offset() > image.storage()->bytes ||
        layout.bytes > image.storage()->bytes - image.byte_offset() ||
        layout.bytes < 64 || static_cast<std::size_t>(rows) > (layout.bytes - 64) / 8 ||
        layout.scales < 64 + static_cast<std::size_t>(rows) * 8 ||
        layout.scales > layout.bytes || layout.scales_bytes > layout.bytes - layout.scales ||
        layout.words > layout.bytes || layout.words_bytes > layout.bytes - layout.words ||
        layout.words < layout.scales + layout.scales_bytes ||
        layout.relative != layout.words + layout.words_bytes ||
        layout.relative > layout.bytes || layout.bytes - layout.relative != 32 ||
        layout.gslog < 0 || layout.gslog > 31)
        throw std::invalid_argument("invalid TQ tensor image or geometry");
    const auto base = reinterpret_cast<std::uintptr_t>(image.data_ptr());
    if (!base || base > std::numeric_limits<std::uintptr_t>::max() - layout.bytes ||
        (base + 64) % 8 || (base + layout.words) % 4)
        throw std::invalid_argument("unaligned TQ tensor image");
    image_ = image;
    storage_ = image.storage();
    byte_offset_ = image.byte_offset();
    device_ = image.device();
    rows_ = rows;
    columns_ = columns;
    layout_ = std::move(layout);
}

inline TqTensor tq_tensor(
    const DenseTensor& image, std::int64_t rows, std::int64_t columns, Tq layout) {
    return TqTensor(image, rows, columns, std::move(layout));
}

inline const Tq& tq_layout(const TqTensor& tensor) { return tensor.layout(); }
inline DenseTensor tq_image(const TqTensor& tensor) { return tensor.image(); }

} // namespace tyloi::encoding
