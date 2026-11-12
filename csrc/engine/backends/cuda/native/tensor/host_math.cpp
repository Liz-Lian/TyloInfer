#include "host.h"

#include <cmath>
#include <limits>

namespace tyloi::cuda::host {
using namespace native_ops_detail;
namespace {

void require_host(const DenseTensor &input) {
  input.require_dense();
  if (!input.is_cpu())
    throw std::invalid_argument("host operation requires a CPU tensor");
  if (input.scalar_type() == kFloat8E4M3FN)
    throw std::invalid_argument("host numeric operation does not support FP8");
}

bool comparison(int operation) {
  return (operation >= 4 && operation <= 7) || operation == 10 ||
         operation == 11;
}

template <typename T> T arithmetic(T a, T b, int op) {
  if constexpr (std::is_same_v<T, std::int64_t>) {
    const auto ua = std::bit_cast<std::uint64_t>(a);
    const auto ub = std::bit_cast<std::uint64_t>(b);
    switch (op) {
    case 0:
      return std::bit_cast<std::int64_t>(ua + ub);
    case 1:
      return std::bit_cast<std::int64_t>(ua - ub);
    case 2:
      return std::bit_cast<std::int64_t>(ua * ub);
    case 3:
      if (b == 0)
        throw std::invalid_argument("integer division by zero");
      if (a == std::numeric_limits<T>::min() && b == -1)
        throw std::overflow_error("integer division overflow");
      return a / b;
    default:
      throw std::invalid_argument(
          "unsupported host integer arithmetic operation");
    }
  }
  switch (op) {
  case 0:
    return a + b;
  case 1:
    return a - b;
  case 2:
    return a * b;
  case 3:
    return a / b;
  case 12:
    return static_cast<T>(std::pow(a, b));
  default:
    throw std::invalid_argument("unsupported host arithmetic operation");
  }
}

template <typename T> bool compare(T a, T b, int op) {
  switch (op) {
  case 4:
    return a < b;
  case 5:
    return a <= b;
  case 6:
    return a > b;
  case 7:
    return a >= b;
  case 10:
    return a == b;
  case 11:
    return a != b;
  default:
    throw std::invalid_argument("unsupported host comparison");
  }
}

bool compare_integer_scalar(std::int64_t integer, double value, int op,
                            bool scalar_first) {
  if (scalar_first) {
    if (op == 4)
      op = 6;
    else if (op == 5)
      op = 7;
    else if (op == 6)
      op = 4;
    else if (op == 7)
      op = 5;
  }
  if (std::isnan(value))
    return op == 11;
  int order;
  if (value >= 9223372036854775808.0)
    order = -1;
  else if (value < -9223372036854775808.0)
    order = 1;
  else {
    const auto integral = static_cast<std::int64_t>(value);
    if (integer < integral)
      order = -1;
    else if (integer > integral)
      order = 1;
    else
      order = value > 0 && value != std::trunc(value)   ? -1
              : value < 0 && value != std::trunc(value) ? 1
                                                        : 0;
  }
  return compare(order, 0, op);
}

template <typename T> T unary_value(T x, int op, T a, T b) {
  switch (op) {
  case 0:
    return std::abs(x);
  case 1:
    return x * x;
  case 2:
    return std::exp(x);
  case 3:
    return std::sqrt(x);
  case 4:
    return std::fmin(std::fmax(x, a), b);
  case 5:
    return std::fmax(x, a);
  case 6:
    return std::fmin(x, a);
  case 7:
    return x - std::floor(x / a) * a;
  case 9:
    return std::sin(x);
  case 10:
    return std::cos(x);
  case 11:
    return T(1) / (T(1) + std::exp(-x));
  case 12:
    return std::tanh(x);
  case 13:
    return std::fmax(x, T(0));
  case 14:
    return T(1) / std::sqrt(x);
  case 15:
    return T(1) / x;
  case 16:
    return std::ceil(x);
  case 17:
    return x > T(20) ? x : std::log1p(std::exp(x));
  case 18:
    return std::log(x);
  case 19:
    return std::log1p(x);
  case 20:
    return std::log2(x);
  case 21:
    return std::exp2(x);
  case 22:
    return x / (T(1) + std::exp(-x));
  case 23:
    return T(0.5) * x * (T(1) + std::erf(x * T(0.7071067811865475244)));
  case 24:
    return std::pow(x, a);
  case 27:
    return T(0.5) * x *
           (T(1) +
            std::tanh(T(0.7978845608028654) * (x + T(0.044715) * x * x * x)));
  default:
    throw std::invalid_argument("unsupported host unary operation");
  }
}

template <typename T>
void binary_values(const TensorView &out, const TensorView &a,
                   const TensorView &b, std::int64_t count, int op) {
  for (std::int64_t i = 0; i < count; ++i) {
    const auto x = load<T>(pointer(a, offset(a, i)), a.scalar_type);
    const auto y = load<T>(pointer(b, offset(b, i)), b.scalar_type);
    auto *destination = pointer(out, i);
    if (comparison(op))
      store(destination, out.scalar_type, compare(x, y, op));
    else if (op == 8 || op == 9) {
      if constexpr (std::is_same_v<T, std::int64_t>) {
        store(destination, out.scalar_type, op == 8 ? x & y : x | y);
      } else {
        throw std::invalid_argument(
            "bitwise operation requires integer tensors");
      }
    } else
      store(destination, out.scalar_type, arithmetic(x, y, op));
  }
}

std::int64_t slice_base(const TensorView &input, const TensorView &output,
                        std::int64_t linear, std::size_t axis, bool keep) {
  std::array<std::int64_t, kMaximumTensorRank> coordinates{};
  for (std::size_t reverse = output.rank; reverse > 0; --reverse) {
    const auto d = reverse - 1;
    coordinates[d] = linear % output.sizes[d];
    linear /= output.sizes[d];
  }
  std::int64_t base = 0;
  std::size_t out_axis = 0;
  for (std::size_t d = 0; d < input.rank; ++d) {
    if (d == axis)
      continue;
    base += coordinates[keep ? d : out_axis++] * input.strides[d];
  }
  return base;
}

template <typename T>
void reduce_values(const TensorView &in, const TensorView &out,
                   std::int64_t count, std::size_t axis, bool keep, int op) {
  for (std::int64_t i = 0; i < count; ++i) {
    const auto base = slice_base(in, out, i, axis, keep);
    T acc = (op == 2 || op == 3) ? std::numeric_limits<T>::lowest() : T(0);
    std::int64_t best = 0;
    bool flag = op == 4;
    for (std::int64_t j = 0; j < in.sizes[axis]; ++j) {
      const auto x =
          load<T>(pointer(in, base + j * in.strides[axis]), in.scalar_type);
      if (op == 0 || op == 1) {
        if constexpr (std::is_same_v<T, std::int64_t>)
          acc = arithmetic(acc, x, 0);
        else
          acc += x;
      }
      if ((op == 2 || op == 3) && x > acc) {
        acc = x;
        best = j;
      }
      if (op == 4)
        flag = flag && x != T(0);
      if (op == 5)
        flag = flag || x != T(0);
    }
    auto *target = pointer(out, i);
    if (op == 3)
      store(target, out.scalar_type, best);
    else if (op == 4 || op == 5)
      store(target, out.scalar_type, flag);
    else {
      if (op == 1)
        acc /= static_cast<T>(in.sizes[axis]);
      store(target, out.scalar_type, acc);
    }
  }
}

template <typename T>
void softmax_values(const TensorView &in, const TensorView &out,
                    std::int64_t rows, std::size_t axis) {
  TensorView reduced = out;
  reduced.sizes[axis] = 1;
  for (std::int64_t row = 0; row < rows; ++row) {
    const auto src = slice_base(in, reduced, row, axis, true);
    const auto dst = slice_base(out, reduced, row, axis, true);
    T maximum = -std::numeric_limits<T>::infinity();
    for (std::int64_t j = 0; j < in.sizes[axis]; ++j) {
      maximum =
          std::fmax(maximum, load<T>(pointer(in, src + j * in.strides[axis]),
                                     in.scalar_type));
    }
    T sum = 0;
    for (std::int64_t j = 0; j < in.sizes[axis]; ++j) {
      sum += std::exp(
          load<T>(pointer(in, src + j * in.strides[axis]), in.scalar_type) -
          maximum);
    }
    for (std::int64_t j = 0; j < in.sizes[axis]; ++j) {
      const auto x =
          load<T>(pointer(in, src + j * in.strides[axis]), in.scalar_type);
      store(pointer(out, dst + j * out.strides[axis]), out.scalar_type,
            std::exp(x - maximum) / sum);
    }
  }
}
} // namespace

DenseTensor unary(const DenseTensor &input, int op, double first,
                  double second) {
  require_host(input);
  const bool boolean = op == 8 || op == 25 || op == 26;
  auto output =
      empty(input.sizes(),
            input.options().dtype(boolean ? kBool : input.scalar_type()));
  const auto in = input.view_descriptor(), out = output.view_descriptor();
  for (std::int64_t i = 0; i < input.numel(); ++i) {
    const auto *src = pointer(in, offset(in, i));
    auto *dst = pointer(out, i);
    if (!floating(in.scalar_type)) {
      const auto x = load<std::int64_t>(src, in.scalar_type);
      if (op == 8)
        store(dst, out.scalar_type, x == 0);
      else if (op == 0)
        store(dst, out.scalar_type,
              x < 0 ? arithmetic(std::int64_t{0}, x, 1) : x);
      else if (op == 1)
        store(dst, out.scalar_type, arithmetic(x, x, 2));
      else if (op == 25)
        store(dst, out.scalar_type, true);
      else if (op == 26)
        store(dst, out.scalar_type, false);
      else if (op == 4 || op == 5 || op == 6) {
        auto result = x;
        if ((op == 4 || op == 5) &&
            compare_integer_scalar(result, first, 4, false)) {
          if (!std::isfinite(first) || first < -9223372036854775808.0 ||
              first >= 9223372036854775808.0)
            throw std::overflow_error("integer clamp bound out of range");
          result = static_cast<std::int64_t>(first);
        }
        if ((op == 4 || op == 6) &&
            compare_integer_scalar(result, op == 4 ? second : first, 6,
                                   false)) {
          const auto bound = op == 4 ? second : first;
          if (!std::isfinite(bound) || bound < -9223372036854775808.0 ||
              bound >= 9223372036854775808.0)
            throw std::overflow_error("integer clamp bound out of range");
          result = static_cast<std::int64_t>(bound);
        }
        store(dst, out.scalar_type, result);
      } else if (op == 7) {
        if (!std::isfinite(first) || std::trunc(first) != first ||
            first < -9223372036854775808.0 || first >= 9223372036854775808.0)
          throw std::invalid_argument(
              "integer remainder requires an integer divisor");
        const auto divisor = static_cast<std::int64_t>(first);
        if (divisor == 0)
          throw std::invalid_argument("integer remainder by zero");
        auto rem =
            x == std::numeric_limits<std::int64_t>::min() && divisor == -1
                ? 0
                : x % divisor;
        if (rem != 0 && ((rem < 0) != (divisor < 0)))
          rem = arithmetic(rem, divisor, 0);
        store(dst, out.scalar_type, rem);
      } else
        throw std::invalid_argument("unsupported host integer unary operation");
    } else if (in.scalar_type == kFloat64) {
      const auto x = load<double>(src, in.scalar_type);
      if (boolean)
        store(dst, out.scalar_type,
              op == 8    ? x == 0
              : op == 25 ? std::isfinite(x)
                         : std::isinf(x) && x < 0);
      else
        store(dst, out.scalar_type, unary_value(x, op, first, second));
    } else {
      const auto x = load<float>(src, in.scalar_type);
      if (boolean)
        store(dst, out.scalar_type,
              op == 8    ? x == 0
              : op == 25 ? std::isfinite(x)
                         : std::isinf(x) && x < 0);
      else
        store(dst, out.scalar_type,
              unary_value(x, op, static_cast<float>(first),
                          static_cast<float>(second)));
    }
  }
  return output;
}

DenseTensor binary(const DenseTensor &left, const DenseTensor &right, int op) {
  require_host(left);
  require_host(right);
  if (left.device() != right.device())
    throw std::invalid_argument("host binary device mismatch");
  const auto type = promote(left.scalar_type(), right.scalar_type());
  const auto shape = broadcast_shape(left, right);
  auto output =
      empty(shape, left.options().dtype(comparison(op) ? kBool : type));
  const auto promoted_left = left.scalar_type() == type ? left : left.to(type);
  const auto promoted_right =
      right.scalar_type() == type ? right : right.to(type);
  const auto a = align_for_broadcast(promoted_left, shape),
             b = align_for_broadcast(promoted_right, shape),
             out = output.view_descriptor();
  if (!floating(type))
    binary_values<std::int64_t>(out, a, b, output.numel(), op);
  else if (type == kFloat64)
    binary_values<double>(out, a, b, output.numel(), op);
  else
    binary_values<float>(out, a, b, output.numel(), op);
  return output;
}

DenseTensor scalar(const DenseTensor &input, double value, int op,
                   bool scalar_first) {
  require_host(input);
  auto output = empty(
      input.sizes(),
      input.options().dtype(comparison(op) ? kBool : input.scalar_type()));
  const auto in = input.view_descriptor(), out = output.view_descriptor();
  for (std::int64_t i = 0; i < input.numel(); ++i) {
    const auto *src = pointer(in, offset(in, i));
    auto *dst = pointer(out, i);
    if (!floating(in.scalar_type) && comparison(op)) {
      store(dst, out.scalar_type,
            compare_integer_scalar(load<std::int64_t>(src, in.scalar_type),
                                   value, op, scalar_first));
    } else if (!floating(in.scalar_type) && std::isfinite(value) &&
               std::trunc(value) == value && value >= -9223372036854775808.0 &&
               value < 9223372036854775808.0) {
      const auto x = load<std::int64_t>(src, in.scalar_type),
                 y = static_cast<std::int64_t>(value);
      const auto a = scalar_first ? y : x, b = scalar_first ? x : y;
      if (comparison(op))
        store(dst, out.scalar_type, compare(a, b, op));
      else if (op == 8 || op == 9)
        store(dst, out.scalar_type, op == 8 ? a & b : a | b);
      else
        store(dst, out.scalar_type, arithmetic(a, b, op));
    } else if (!floating(in.scalar_type)) {
      throw std::invalid_argument(
          "integer arithmetic requires a representable integer scalar");
    } else if (in.scalar_type == kFloat64) {
      const auto x = load<double>(src, in.scalar_type),
                 a = scalar_first ? value : x, b = scalar_first ? x : value;
      if (comparison(op))
        store(dst, out.scalar_type, compare(a, b, op));
      else if (op == 8 || op == 9)
        throw std::invalid_argument(
            "bitwise operation requires integer tensors");
      else
        store(dst, out.scalar_type, arithmetic(a, b, op));
    } else {
      const auto x = load<float>(src, in.scalar_type),
                 y = static_cast<float>(value);
      const auto a = scalar_first ? y : x, b = scalar_first ? x : y;
      if (comparison(op))
        store(dst, out.scalar_type, compare(a, b, op));
      else if (op == 8 || op == 9)
        throw std::invalid_argument(
            "bitwise operation requires integer tensors");
      else
        store(dst, out.scalar_type, arithmetic(a, b, op));
    }
  }
  return output;
}

DenseTensor reduce(const DenseTensor &input, std::int64_t dimension, bool keep,
                   int op) {
  require_host(input);
  if (input.dim() == 0 || op < 0 || op > 5 ||
      (op == 1 && !floating(input.scalar_type()))) {
    throw std::invalid_argument("invalid host reduction");
  }
  const auto axis = normalize_dimension(dimension, input.dim());
  if (input.size(axis) == 0)
    throw std::invalid_argument("cannot reduce an empty dimension");
  auto shape = input.sizes().vec();
  if (keep)
    shape[axis] = 1;
  else
    shape.erase(shape.begin() + static_cast<std::ptrdiff_t>(axis));
  const auto type = op == 3 || (op == 0 && !floating(input.scalar_type()))
                        ? kInt64
                        : (op == 4 || op == 5 ? kBool : input.scalar_type());
  auto output = empty(shape, input.options().dtype(type));
  const auto in = input.view_descriptor(), out = output.view_descriptor();
  if (!floating(input.scalar_type()))
    reduce_values<std::int64_t>(in, out, output.numel(), axis, keep, op);
  else if (input.scalar_type() == kFloat64)
    reduce_values<double>(in, out, output.numel(), axis, keep, op);
  else
    reduce_values<float>(in, out, output.numel(), axis, keep, op);
  return output;
}

DenseTensor where(const DenseTensor &condition, const DenseTensor &yes,
                  const DenseTensor &no) {
  require_host(condition);
  require_host(yes);
  require_host(no);
  if (condition.device() != yes.device() || yes.device() != no.device())
    throw std::invalid_argument("host where device mismatch");
  auto shape = broadcast_shape(yes, no);
  shape = broadcast_shape(yes.expand(shape), condition);
  const auto type = promote(yes.scalar_type(), no.scalar_type());
  auto output = empty(shape, yes.options().dtype(type));
  const auto promoted_yes = yes.scalar_type() == type ? yes : yes.to(type);
  const auto promoted_no = no.scalar_type() == type ? no : no.to(type);
  const auto c = align_for_broadcast(condition, shape),
             a = align_for_broadcast(promoted_yes, shape);
  const auto b = align_for_broadcast(promoted_no, shape),
             out = output.view_descriptor();
  for (std::int64_t i = 0; i < output.numel(); ++i) {
    const auto &src =
        load<double>(pointer(c, offset(c, i)), c.scalar_type) != 0 ? a : b;
    const auto *value = pointer(src, offset(src, i));
    if (!floating(type))
      store(pointer(out, i), type, load<std::int64_t>(value, src.scalar_type));
    else if (type == kFloat64)
      store(pointer(out, i), type, load<double>(value, src.scalar_type));
    else
      store(pointer(out, i), type, load<float>(value, src.scalar_type));
  }
  return output;
}

DenseTensor softmax(const DenseTensor &input, std::int64_t dimension) {
  require_host(input);
  if (!floating(input.scalar_type()))
    throw std::invalid_argument("host softmax requires floating dtype");
  const auto axis = normalize_dimension(dimension, input.dim());
  auto output = empty(input.sizes(), input.options());
  const auto extent = input.size(axis);
  if (extent == 0)
    return output;
  const auto in = input.view_descriptor(), out = output.view_descriptor();
  if (input.scalar_type() == kFloat64)
    softmax_values<double>(in, out, input.numel() / extent, axis);
  else
    softmax_values<float>(in, out, input.numel() / extent, axis);
  return output;
}
} // namespace tyloi::cuda::host
