#include "mlx_gguf_rows.h"
#include "mlx_kernel_prepare.h"

#include <climits>
#include <stdexcept>

namespace mfq::metal {

void MlxGgufRowBatch::append(std::string_view dtype, int width, std::span<const std::uint8_t> bytes) {
    const auto stride = gguf_packed_row_bytes(dtype, width);
    if (bytes.size() != stride) throw std::invalid_argument("GGUF selected row length differs");
    if (bytes_.empty()) { dtype_ = dtype; width_ = width; row_bytes_ = stride; }
    if (dtype_ != dtype || width_ != width) throw std::invalid_argument("GGUF selected row formats differ");
    bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
}
void MlxGgufRowBatch::append_batch(const MlxGgufRowBatch& batch) {
    if (batch.bytes_.empty()) return;
    if (this == &batch) throw std::invalid_argument("GGUF selected row batch cannot append itself");
    if (bytes_.empty()) { dtype_ = batch.dtype_; width_ = batch.width_; row_bytes_ = batch.row_bytes_; }
    if (dtype_ != batch.dtype_ || width_ != batch.width_)
        throw std::invalid_argument("GGUF selected row batch formats differ");
    bytes_.insert(bytes_.end(), batch.bytes_.begin(), batch.bytes_.end());
}

mlx::core::array MlxGgufRowBatch::decode() const {
    if (!row_bytes_ || bytes_.empty() || bytes_.size() / row_bytes_ > INT_MAX)
        throw std::invalid_argument("GGUF selected row batch is empty or too large");
    const int count = static_cast<int>(bytes_.size() / row_bytes_);
    return MlxGgufWeight::from_bytes(dtype_, bytes_, {count, width_}).dequantize();
}

MlxMappedGgufRows::MlxMappedGgufRows(std::span<const std::uint8_t> bytes,
        std::string dtype, std::int64_t rows, int width)
    : bytes_(bytes), dtype_(std::move(dtype)), rows_(rows), width_(width),
      row_bytes_(gguf_packed_row_bytes(dtype_, width_)) {
    if (rows <= 0 || std::uint64_t(rows) > SIZE_MAX / row_bytes_ ||
        std::uint64_t(rows) * row_bytes_ != bytes.size())
        throw std::invalid_argument("GGUF mapped row geometry differs");
    if (auto* preparation = MlxKernelPreparation::current()) {
        preparation->collect([&] {
            const std::vector<std::uint8_t> packed(row_bytes_, 0);
            preparation->add(MlxGgufWeight::from_bytes(dtype_, packed, {1, width_}).dequantize());
        });
    }
}

void MlxMappedGgufRows::append_row(std::int64_t row, MlxGgufRowBatch& batch) const {
    if (row < 0 || row >= rows_) throw std::out_of_range("GGUF selected row is outside the table");
    batch.append(dtype_, width_, bytes_.subspan(std::size_t(row) * row_bytes_, row_bytes_));
}
} // namespace mfq::metal
