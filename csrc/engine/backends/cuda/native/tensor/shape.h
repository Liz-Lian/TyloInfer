#pragma once

#include "tensor.h"

namespace tyloi::cuda::native_ops_detail {

inline std::size_t normalize_dimension(std::int64_t dimension,
                                       std::size_t rank) {
  const auto normalized =
      dimension < 0 ? dimension + static_cast<std::int64_t>(rank) : dimension;
  if (normalized < 0 || normalized >= static_cast<std::int64_t>(rank)) {
    throw std::out_of_range("tensor dimension is out of range");
  }
  return static_cast<std::size_t>(normalized);
}

inline bool floating(ScalarType type) {
  return type == kFloat16 || type == kBFloat16 || type == kFloat32 ||
         type == kFloat64;
}

inline ScalarType promote(ScalarType left, ScalarType right) {
  if (left == right)
    return left;
  if (left == kFloat64 || right == kFloat64)
    return kFloat64;
  if (left == kFloat32 || right == kFloat32)
    return kFloat32;
  if ((left == kFloat16 && right == kBFloat16) ||
      (left == kBFloat16 && right == kFloat16))
    return kFloat32;
  if (left == kBFloat16 || right == kBFloat16)
    return kBFloat16;
  if (left == kFloat16 || right == kFloat16)
    return kFloat16;
  if (left == kInt64 || right == kInt64)
    return kInt64;
  if (left == kInt32 || right == kInt32)
    return kInt32;
  if (left == kInt16 || right == kInt16)
    return kInt16;
  if (left == kInt8 || right == kInt8)
    return kInt8;
  if (left == kUInt8 || right == kUInt8)
    return kUInt8;
  return kBool;
}

inline TensorView
align_for_broadcast(const DenseTensor &input,
                    std::span<const std::int64_t> output_shape) {
  if (input.dim() > static_cast<std::int64_t>(output_shape.size())) {
    throw std::invalid_argument("broadcast rank mismatch");
  }
  auto result = input.view_descriptor();
  std::array<std::int64_t, kMaximumTensorRank> sizes{};
  std::array<std::int64_t, kMaximumTensorRank> strides{};
  const auto offset =
      output_shape.size() - static_cast<std::size_t>(input.dim());
  for (std::size_t dimension = 0; dimension < output_shape.size();
       ++dimension) {
    sizes[dimension] = output_shape[dimension];
    if (dimension < offset) {
      strides[dimension] = 0;
      continue;
    }
    const auto source_dimension = dimension - offset;
    const auto source_extent =
        input.size(static_cast<std::int64_t>(source_dimension));
    if (source_extent != output_shape[dimension] && source_extent != 1) {
      throw std::invalid_argument("tensor dimensions cannot be broadcast");
    }
    strides[dimension] =
        source_extent == 1
            ? 0
            : input.stride(static_cast<std::int64_t>(source_dimension));
  }
  result.rank = static_cast<std::uint8_t>(output_shape.size());
  result.sizes = sizes;
  result.strides = strides;
  return result;
}

inline std::vector<std::int64_t> broadcast_shape(const DenseTensor &left,
                                                 const DenseTensor &right) {
  const auto rank = static_cast<std::size_t>(std::max(left.dim(), right.dim()));
  if (rank > kMaximumTensorRank)
    throw std::invalid_argument("broadcast rank exceeds ABI");
  std::vector<std::int64_t> result(rank, 1);
  for (std::size_t reverse = 0; reverse < rank; ++reverse) {
    const auto left_dimension =
        left.dim() - 1 - static_cast<std::int64_t>(reverse);
    const auto right_dimension =
        right.dim() - 1 - static_cast<std::int64_t>(reverse);
    const auto l = left_dimension >= 0 ? left.size(left_dimension) : 1;
    const auto r = right_dimension >= 0 ? right.size(right_dimension) : 1;
    if (l != r && l != 1 && r != 1) {
      throw std::invalid_argument("tensor dimensions cannot be broadcast");
    }
    result[rank - 1 - reverse] = l == 1 ? r : l;
  }
  return result;
}

inline bool same_shape(const DenseTensor &left, const DenseTensor &right) {
  if (left.dim() != right.dim()) {
    return false;
  }
  for (std::int64_t dimension = 0; dimension < left.dim(); ++dimension) {
    if (left.size(dimension) != right.size(dimension)) {
      return false;
    }
  }
  return true;
}

} // namespace tyloi::cuda::native_ops_detail
