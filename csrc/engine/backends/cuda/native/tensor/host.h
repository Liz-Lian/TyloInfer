#pragma once

#include "shape.h"

#include <bit>

namespace tyloi::cuda::host {

inline float decode_float16(std::uint16_t bits) {
  const auto sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
  auto exponent = static_cast<std::uint32_t>((bits >> 10) & 0x1fu);
  auto mantissa = static_cast<std::uint32_t>(bits & 0x03ffu);
  std::uint32_t value = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      value = sign;
    } else {
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      value = sign | (exponent << 23) | ((mantissa & 0x03ffu) << 13);
    }
  } else if (exponent == 0x1fu) {
    value = sign | 0x7f800000u | (mantissa << 13);
  } else {
    value = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  return std::bit_cast<float>(value);
}

/** Resolve a logical row-major index to an element offset in a host view. */
inline std::int64_t offset(const TensorView &view, std::int64_t linear) {
  std::int64_t result = 0;
  for (std::size_t reverse = view.rank; reverse > 0; --reverse) {
    const auto axis = reverse - 1;
    result += (linear % view.sizes[axis]) * view.strides[axis];
    linear /= view.sizes[axis];
  }
  return result;
}

/** Read a host scalar directly into the selected arithmetic type. */
template <typename Value> Value load(const void *pointer, ScalarType type) {
  switch (type) {
  case kBool:
    return static_cast<Value>(*static_cast<const bool *>(pointer));
  case kUInt8:
    return static_cast<Value>(*static_cast<const std::uint8_t *>(pointer));
  case kInt8:
    return static_cast<Value>(*static_cast<const std::int8_t *>(pointer));
  case kInt16:
    return static_cast<Value>(*static_cast<const std::int16_t *>(pointer));
  case kInt32:
    return static_cast<Value>(*static_cast<const std::int32_t *>(pointer));
  case kInt64:
    return static_cast<Value>(*static_cast<const std::int64_t *>(pointer));
  case kFloat16:
    return static_cast<Value>(
        decode_float16(*static_cast<const std::uint16_t *>(pointer)));
  case kBFloat16:
    return static_cast<Value>(std::bit_cast<float>(
        static_cast<std::uint32_t>(*static_cast<const std::uint16_t *>(pointer))
        << 16));
  case kFloat32:
    return static_cast<Value>(*static_cast<const float *>(pointer));
  case kFloat64:
    return static_cast<Value>(*static_cast<const double *>(pointer));
  case kFloat8E4M3FN:
    throw std::invalid_argument("CPU FP8 conversion is format-specific");
  }
  throw std::invalid_argument("unknown CPU tensor dtype");
}

/** Reject undefined floating-to-integer conversions before storing a host
 * scalar. */
template <typename Output, typename Value> Output convert(Value value) {
  if constexpr (std::is_floating_point_v<Value> && std::is_integral_v<Output> &&
                !std::is_same_v<Output, bool>) {
    const auto truncated = std::trunc(value);
    constexpr auto digits = std::numeric_limits<Output>::digits;
    const auto upper = std::ldexp(Value(1), digits);
    const auto lower = std::is_signed_v<Output> ? -upper : Value(0);
    if (!std::isfinite(truncated) || truncated < lower || truncated >= upper)
      throw std::out_of_range("host scalar is outside its integer dtype");
  }
  return static_cast<Output>(value);
}

/** Store a host scalar, retaining integer values and native reduced-precision
 * rounding. */
template <typename Value>
void store(void *pointer, ScalarType type, Value value) {
  switch (type) {
  case kBool:
    *static_cast<bool *>(pointer) = value != 0;
    return;
  case kUInt8:
    *static_cast<std::uint8_t *>(pointer) = convert<std::uint8_t>(value);
    return;
  case kInt8:
    *static_cast<std::int8_t *>(pointer) = convert<std::int8_t>(value);
    return;
  case kInt16:
    *static_cast<std::int16_t *>(pointer) = convert<std::int16_t>(value);
    return;
  case kInt32:
    *static_cast<std::int32_t *>(pointer) = convert<std::int32_t>(value);
    return;
  case kInt64:
    *static_cast<std::int64_t *>(pointer) = convert<std::int64_t>(value);
    return;
  case kFloat16:
    *static_cast<std::uint16_t *>(pointer) =
        encode_float16(static_cast<float>(value));
    return;
  case kBFloat16:
    *static_cast<std::uint16_t *>(pointer) =
        encode_bfloat16(static_cast<float>(value));
    return;
  case kFloat32:
    *static_cast<float *>(pointer) = static_cast<float>(value);
    return;
  case kFloat64:
    *static_cast<double *>(pointer) = static_cast<double>(value);
    return;
  case kFloat8E4M3FN:
    throw std::invalid_argument("CPU FP8 conversion is format-specific");
  }
  throw std::invalid_argument("unknown CPU tensor dtype");
}

/** Return the address of an element offset from this view's existing data
 * origin. */
inline void *pointer(const TensorView &view, std::int64_t index) {
  return static_cast<std::byte *>(view.data) +
         index * scalar_size(view.scalar_type);
}

/** Execute the native elementwise operation on host storage. */
DenseTensor unary(const DenseTensor &input, int operation, double first,
                  double second);
/** Broadcast and execute a native binary operation without moving devices. */
DenseTensor binary(const DenseTensor &left, const DenseTensor &right,
                   int operation);
/** Execute scalar arithmetic or comparison on host storage. */
DenseTensor scalar(const DenseTensor &input, double value, int operation,
                   bool scalar_first);
/** Reduce a host view using the native operation and dtype contract. */
DenseTensor reduce(const DenseTensor &input, std::int64_t dimension, bool keep,
                   int operation);
/** Select broadcast values with a host boolean condition. */
DenseTensor where(const DenseTensor &condition, const DenseTensor &yes,
                  const DenseTensor &no);
/** Compute stable softmax along a host dimension. */
DenseTensor softmax(const DenseTensor &input, std::int64_t dimension);
/** Multiply floating host tensors, including vectors, views, and broadcast
 * batches. */
DenseTensor matmul(const DenseTensor &left, const DenseTensor &right);
/** Select rows of a host view using exact integer indices. */
DenseTensor index_select(const DenseTensor &input, std::int64_t dimension,
                         const DenseTensor &indices);
/** Gather host elements using exact integer indices. */
DenseTensor gather(const DenseTensor &input, std::int64_t dimension,
                   const DenseTensor &indices);
/** Repeat a host view along each supplied dimension. */
DenseTensor repeat(const DenseTensor &input,
                   std::span<const std::int64_t> repeats);
/** Repeat each host element along one dimension. */
DenseTensor repeat_interleave(const DenseTensor &input, std::int64_t repeats,
                              std::int64_t dimension);
/** Copy complete indexed host slices after validating geometry and indices. */
void index_copy(DenseTensor &destination, std::int64_t dimension,
                const DenseTensor &indices, const DenseTensor &source);
/** Fill indexed host slices without converting integer indices to floating
 * point. */
void index_fill(DenseTensor &destination, std::int64_t dimension,
                const DenseTensor &indices, double value);
/** Fill a host view where a broadcast mask is true. */
void masked_fill(DenseTensor &destination, const DenseTensor &mask,
                 double value);
/** Pack the host elements selected by a broadcast mask. */
DenseTensor masked_select(const DenseTensor &input, const DenseTensor &mask);

} // namespace tyloi::cuda::host
