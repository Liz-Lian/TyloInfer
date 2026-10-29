#include "mlx_affine_rows.h"
#include "mlx_affine.h"

#include <mlx/allocator.h>
#include <climits>
#include <cstring>
#include <stdexcept>

namespace mfq::metal {
void MlxAffineRowBatch::append(const mfq::MlxAffineLayout& layout, std::span<const std::uint8_t> values,
    std::span<const std::uint8_t> scales, std::span<const std::uint8_t> biases) {
    if (values_.empty()) layout_ = layout;
    if (layout_.bits != layout.bits || layout_.group_size != layout.group_size ||
        layout_.scale_dtype != layout.scale_dtype || layout_.shape.back() != layout.shape.back() ||
        values.size() != layout.values_bytes / layout.shape[0] ||
        scales.size() != layout.scales_bytes / layout.shape[0] || biases.size() != scales.size())
        throw std::invalid_argument("MLX selected row formats or extents differ");
    values_.insert(values_.end(), values.begin(), values.end());
    scales_.insert(scales_.end(), scales.begin(), scales.end());
    biases_.insert(biases_.end(), biases.begin(), biases.end());
}
void MlxAffineRowBatch::append_batch(const MlxAffineRowBatch& batch) {
    if (batch.values_.empty()) return;
    if (this == &batch) throw std::invalid_argument("MLX selected row batch cannot append itself");
    if (values_.empty()) layout_ = batch.layout_;
    if (layout_.bits != batch.layout_.bits || layout_.group_size != batch.layout_.group_size ||
        layout_.scale_dtype != batch.layout_.scale_dtype || layout_.shape.back() != batch.layout_.shape.back())
        throw std::invalid_argument("MLX selected row batch formats differ");
    values_.insert(values_.end(), batch.values_.begin(), batch.values_.end());
    scales_.insert(scales_.end(), batch.scales_.begin(), batch.scales_.end());
    biases_.insert(biases_.end(), batch.biases_.begin(), batch.biases_.end());
}
mlx::core::array MlxAffineRowBatch::decode(mlx::core::Dtype output_dtype) const {
    using namespace mlx::core;
    const auto stride = layout_.shape.empty() ? 0 : layout_.values_bytes / layout_.shape[0];
    if (!stride || values_.empty() || values_.size() / stride > INT_MAX)
        throw std::invalid_argument("MLX selected row batch is empty or too large");
    const int count = int(values_.size() / stride), width = int(layout_.shape.back());
    const auto dtype = layout_.scale_dtype == "BF16" ? bfloat16 : layout_.scale_dtype == "F16" ? float16 : float32;
    const auto field = [&](const auto& bytes, Shape shape, Dtype dtype) {
        array value(allocator::malloc(bytes.size()), std::move(shape), dtype);
        std::memcpy(value.data<std::uint8_t>(), bytes.data(), bytes.size()); return value;
    };
    return MlxAffineWeight::from_arrays(field(values_, {count, width / 32 * layout_.bits}, uint32),
        field(scales_, {count, width / layout_.group_size}, dtype),
        field(biases_, {count, width / layout_.group_size}, dtype), layout_.bits, layout_.group_size).dequantize(output_dtype);
}
MlxMappedAffineRows::MlxMappedAffineRows(const MfqContainer& model, const std::string& name) {
    auto prefix = model.read_range(name, 0, 8);
    prefix = model.read_range(name, 0,
        mfq::MlxAffineLayout::prefix_size(reinterpret_cast<const std::byte*>(prefix.data()), prefix.size()));
    layout_ = mfq::MlxAffineLayout::parse(reinterpret_cast<const std::byte*>(prefix.data()), prefix.size());
    if (layout_.shape.size() != 2 || layout_.shape.back() > INT_MAX || layout_.total_bytes() != model.record(name).nbytes)
        throw std::invalid_argument("MLX mapped row geometry differs");
    values_ = model.map_record_range(name, layout_.prefix_bytes, layout_.values_bytes, true);
    scales_ = model.map_record_range(name, layout_.scales_offset(), layout_.scales_bytes, true);
    biases_ = model.map_record_range(name, layout_.biases_offset(), layout_.scales_bytes, true);
    value_stride_ = layout_.values_bytes / layout_.shape[0]; scale_stride_ = layout_.scales_bytes / layout_.shape[0];
}
void MlxMappedAffineRows::append_row(std::int64_t row, MlxAffineRowBatch& batch) const {
    if (row < 0 || row >= rows()) throw std::out_of_range("MLX selected row is outside the table");
    batch.append(layout_, values_.view().subspan(row * value_stride_, value_stride_),
        scales_.view().subspan(row * scale_stride_, scale_stride_), biases_.view().subspan(row * scale_stride_, scale_stride_));
}

} // namespace mfq::metal
