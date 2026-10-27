#include "mlx_affine.h"
#include "mlx_affine_staged.h"
#include "mlx_moe_ops.h"
#include "mfq_container.h"
#include "mlx_weight_residency.h"
#include "tyloinfer/mlx_affine.h"

#include <mlx/allocator.h>
#include <climits>
#include <stdexcept>

namespace mfq::metal {
namespace {
using namespace mlx::core;
std::string dtype_name(Dtype dtype) {
    if (dtype == float16) return "F16";
    if (dtype == bfloat16) return "BF16";
    if (dtype == float32) return "F32";
    throw std::invalid_argument("MLX affine scales must be floating point");
}
void check_input(const array& input, int width) {
    if (input.ndim() < 1 || input.shape(-1) != width ||
        (input.dtype() != float16 && input.dtype() != bfloat16 && input.dtype() != float32))
        throw std::invalid_argument("MLX affine projection input mismatch");
}
}

MlxAffineWeight::MlxAffineWeight(array values, array scales, array biases, int bits, int group_size, int width)
    : values_(std::move(values)), scales_(std::move(scales)), biases_(std::move(biases)),
      bits_(bits), group_size_(group_size), width_(width) {}

MlxAffineWeight MlxAffineWeight::from_arrays(array values, array scales, array biases, int bits, int group_size) {
    if (values.dtype() != uint32 || (values.ndim() != 2 && values.ndim() != 3) || bits < 1 ||
        std::int64_t(values.shape(-1)) * 32 % bits || biases.dtype() != scales.dtype())
        throw std::invalid_argument("invalid MLX affine packed/scales/biases arrays");
    mfq::MlxAffineLayout layout;
    layout.shape.assign(values.shape().begin(), values.shape().end());
    layout.shape.back() = layout.shape.back() * 32 / bits;
    layout.bits = bits; layout.group_size = group_size; layout.scale_dtype = dtype_name(scales.dtype());
    layout.validate();
    if (layout.shape.back() > INT_MAX) throw std::invalid_argument("MLX affine row exceeds runtime limits");
    auto shape = values.shape(); shape.back() = int(layout.shape.back() / group_size);
    if (scales.shape() != shape || biases.shape() != shape ||
        !values.flags().row_contiguous || !scales.flags().row_contiguous || !biases.flags().row_contiguous)
        throw std::invalid_argument("MLX affine auxiliary array shape/layout mismatch");
    eval(values, scales, biases);
    MlxWeightResidency::track(values); MlxWeightResidency::track(scales); MlxWeightResidency::track(biases);
    return MlxAffineWeight(std::move(values), std::move(scales), std::move(biases), bits, group_size, int(layout.shape.back()));
}
MlxAffineWeight MlxAffineWeight::load(const MfqContainer& model, const std::string& name) {
    if (model.record(name).dtype != "MLX-AFFINE") throw std::invalid_argument("tensor is not MLX affine");
    auto prefix = model.read_range(name, 0, 8);
    const auto size = mfq::MlxAffineLayout::prefix_size(reinterpret_cast<const std::byte*>(prefix.data()), prefix.size());
    prefix = model.read_range(name, 0, size);
    const auto layout = mfq::MlxAffineLayout::parse(reinterpret_cast<const std::byte*>(prefix.data()), prefix.size());
    if (layout.total_bytes() != model.record(name).nbytes)
        throw std::invalid_argument("MLX affine tensor payload length mismatch");
    Shape shape;
    for (const auto dimension : layout.shape) {
        if (dimension > INT_MAX) throw std::invalid_argument("MLX affine tensor dimension exceeds runtime limits");
        shape.push_back(int(dimension));
    }
    const auto read = [&](Shape field_shape, Dtype dtype, std::uint64_t offset, std::size_t bytes) {
        array result(allocator::malloc(bytes), std::move(field_shape), dtype);
        model.read_range_into(name, offset, {reinterpret_cast<std::byte*>(result.data<std::uint8_t>()), bytes});
        return result;
    };
    auto packed_shape = shape; packed_shape.back() = shape.back() / 32 * layout.bits;
    auto scale_shape = shape; scale_shape.back() /= layout.group_size;
    const auto dtype = layout.scale_dtype == "BF16" ? bfloat16 : layout.scale_dtype == "F16" ? float16 : float32;
    auto result = from_arrays(read(packed_shape, uint32, layout.prefix_bytes, layout.values_bytes),
        read(scale_shape, dtype, layout.scales_offset(), layout.scales_bytes),
        read(scale_shape, dtype, layout.biases_offset(), layout.scales_bytes), layout.bits, layout.group_size);
    model.record_prepared(name);
    return result;
}
std::optional<MlxAffineWeight> MlxAffineWeight::join_rows(const MlxAffineWeight& first, const MlxAffineWeight& second) {
    if (first.bits_ != second.bits_ || first.group_size_ != second.group_size_ || first.width_ != second.width_ ||
        first.scales_.dtype() != second.scales_.dtype() || first.values_.ndim() != second.values_.ndim() ||
        first.expert_count() != second.expert_count()) return std::nullopt;
    return from_arrays(concatenate({first.values_, second.values_}, -2),
        concatenate({first.scales_, second.scales_}, -2), concatenate({first.biases_, second.biases_}, -2),
        first.bits_, first.group_size_);
}
array MlxAffineWeight::matmul(const array& input) const {
    check_input(input, width_);
    if (values_.ndim() != 2) throw std::invalid_argument("MLX expert matrix requires routed projection");
    auto result = quantized_matmul(astype(input, scales_.dtype()), values_, scales_, biases_,
        true, group_size_, bits_, "affine");
    return result.dtype() == input.dtype() ? result : astype(result, input.dtype());
}
array MlxAffineWeight::dequantize(Dtype dtype) const {
    return mlx::core::dequantize(values_, scales_, biases_, group_size_, bits_, "affine", std::nullopt, dtype);
}
array MlxAffineWeight::embedding(const array& ids, Dtype dtype) const {
    if (values_.ndim() != 2) throw std::invalid_argument("MLX embedding requires a matrix");
    auto shape = ids.shape(); shape.push_back(width_);
    if (!ids.size()) return zeros(shape, dtype);
    auto index = astype(ids, int32);
    auto safe = clip(index, array(0, int32), array(output_size() - 1, int32));
    auto packed = take(values_, safe, 0), scales = take(scales_, safe, 0), biases = take(biases_, safe, 0);
    auto result = mlx::core::dequantize(packed, scales, biases, group_size_, bits_, "affine", std::nullopt, dtype);
    return where(expand_dims(logical_and(index >= 0, index < output_size()), -1), result, zeros(shape, dtype));
}
MlxAffineRoutePlan::MlxAffineRoutePlan(const array& ids, int experts)
    : MlxAffineRoutePlan(ids, experts, false) {}

MlxAffineRoutePlan MlxAffineRoutePlan::from_topk(const array& ids, int experts) {
    return MlxAffineRoutePlan(ids, experts, true);
}

MlxAffineRoutePlan::MlxAffineRoutePlan(const array& ids, int experts, bool valid)
    : ids_(ids), tokens_(ids.ndim() == 2 ? ids.shape(0) : 0),
      routes_(ids.ndim() == 2 ? ids.shape(1) : 0), experts_(experts) {
    if (ids.ndim() != 2 || (ids.dtype() != int32 && ids.dtype() != uint32) || routes_ < 1 || experts < 1)
        throw std::invalid_argument("MLX route plan requires rank-two integer expert IDs");
    if (!valid) {
        valid_ = logical_and(ids >= 0, ids < experts);
        ids_ = clip(astype(ids, int32), array(0, int32), array(experts - 1, int32));
    }
    ids_ = astype(ids_, uint32);
    if (ids.size() >= 64) {
        order_ = astype(argsort(reshape(ids_, {-1})), uint32);
        inverse_ = astype(argsort(*order_), uint32);
        ids_ = take(reshape(ids_, {-1}), *order_, 0);
    }
}

array MlxAffineRoutePlan::restore(const array& output) const {
    const int width = output.shape(-1);
    auto result = order_ ? take(reshape(output, {-1, width}), *inverse_, 0) : output;
    result = reshape(result, {tokens_, routes_, width});
    return valid_ ? where(expand_dims(*valid_, -1), result, zeros(result.shape(), result.dtype())) : result;
}

array MlxAffineWeight::routed_matmul(const array& input, const array& ids, bool shared_input) const {
    const MlxAffineRoutePlan plan(ids, expert_count());
    return plan.restore(routed_matmul(input, plan, shared_input));
}

array MlxAffineWeight::routed_swiglu(const array& input, const MlxAffineRoutePlan& plan) const {
    check_input(input, width_);
    if (values_.ndim() != 3 || plan.experts_ != expert_count() || output_size() % 2 ||
        input.size() / width_ != std::size_t(plan.tokens_))
        throw std::invalid_argument("MLX routed Gate/Up shape mismatch");
    if (auto result = MlxAffineStaged::project(*this, input, plan, true, false, true)) return *result;
    return moe_swiglu_split(routed_matmul(input, plan, true));
}

array MlxAffineWeight::routed_matmul(const array& input, const MlxAffineRoutePlan& plan,
    bool shared_input, bool input_in_route_order) const {
    check_input(input, width_);
    if (values_.ndim() != 3 || plan.experts_ != expert_count() ||
        input.size() / width_ != std::size_t(plan.tokens_) * (shared_input ? 1 : plan.routes_))
        throw std::invalid_argument("MLX routed projection shape mismatch");
    const Shape shape{plan.tokens_, plan.routes_, output_size()};
    if (!plan.tokens_) return zeros(shape, input.dtype());
    if (auto result = MlxAffineStaged::project(*this, input, plan, shared_input, input_in_route_order))
        return *result;
    auto x = astype(input, scales_.dtype());
    if (plan.order_) {
        x = reshape(x, {-1, 1, width_});
        if (shared_input) x = take(x, floor_divide(*plan.order_, array(plan.routes_, uint32)), 0);
        else if (!input_in_route_order) x = take(x, *plan.order_, 0);
    } else x = reshape(x, {plan.tokens_, shared_input ? 1 : plan.routes_, 1, width_});
    auto result = gather_qmm(x, values_, scales_, biases_, std::nullopt, plan.ids_,
        true, group_size_, bits_, "affine", std::nullopt, plan.order_.has_value());
    result = astype(result, input.dtype());
    return reshape(result, plan.order_ ? Shape{plan.tokens_ * plan.routes_, 1, output_size()} : shape);
}

} // namespace mfq::metal
