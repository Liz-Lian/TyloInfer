#include "host.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tyloi::cuda::host {
namespace {
using native_ops_detail::align_for_broadcast;
using native_ops_detail::normalize_dimension;

/** Require a dense CPU tensor before accessing host storage. */
void cpu(const DenseTensor &tensor) {
  tensor.require_dense();
  if (!tensor.is_cpu())
    throw std::invalid_argument("host indexing requires CPU tensors");
}

/** Require host operands to refer to the same device. */
void same_device(const DenseTensor &left, const DenseTensor &right) {
  cpu(right);
  if (left.device() != right.device())
    throw std::invalid_argument("host indexing device mismatch");
}

/** Check a nonnegative product before using it as tensor geometry. */
std::int64_t product(std::int64_t left, std::int64_t right) {
  if (left < 0 || right < 0 ||
      (right && left > std::numeric_limits<std::int64_t>::max() / right))
    throw std::overflow_error("host indexing shape overflow");
  return left * right;
}

/** Validate the output element and byte counts before allocating storage. */
void geometry(const std::vector<std::int64_t> &shape,
              std::size_t element_size) {
  std::int64_t count = 1;
  for (const auto size : shape)
    count = product(count, size);
  product(count, static_cast<std::int64_t>(element_size));
}

/** Validate the entire index tensor and retain exact integer values. */
std::vector<std::int64_t> indices(const DenseTensor &tensor,
                                  std::int64_t extent, bool vector) {
  cpu(tensor);
  if ((vector && tensor.dim() != 1) ||
      (tensor.scalar_type() != kInt32 && tensor.scalar_type() != kInt64))
    throw std::invalid_argument(
        "indices must be an int32 or int64 tensor of the required rank");
  const auto view = tensor.view_descriptor();
  std::vector<std::int64_t> result;
  result.reserve(static_cast<std::size_t>(tensor.numel()));
  for (std::int64_t i = 0; i < tensor.numel(); ++i) {
    const auto *p = pointer(view, offset(view, i));
    const auto value =
        tensor.scalar_type() == kInt32
            ? static_cast<std::int64_t>(*static_cast<const std::int32_t *>(p))
            : *static_cast<const std::int64_t *>(p);
    if (value < 0 || value >= extent)
      throw std::out_of_range("host index is out of bounds");
    result.push_back(value);
  }
  return result;
}

/** Convert a logical row-major index to coordinates in a shape. */
std::vector<std::int64_t> coordinates(std::int64_t linear,
                                      const std::vector<std::int64_t> &shape) {
  std::vector<std::int64_t> result(shape.size());
  for (std::size_t i = shape.size(); i > 0; --i) {
    result[i - 1] = linear % shape[i - 1];
    linear /= shape[i - 1];
  }
  return result;
}

/** Resolve coordinates using the strides of the original view. */
std::int64_t at(const TensorView &view,
                const std::vector<std::int64_t> &coord) {
  std::int64_t result = 0;
  for (std::size_t i = 0; i < coord.size(); ++i)
    result += coord[i] * view.strides[i];
  return result;
}

/** Snapshot a logical tensor so writes cannot change an aliased source. */
std::vector<std::byte> snapshot(const DenseTensor &input) {
  const auto bytes =
      product(input.numel(), static_cast<std::int64_t>(input.element_size()));
  std::vector<std::byte> result(static_cast<std::size_t>(bytes));
  const auto view = input.view_descriptor();
  for (std::int64_t i = 0; i < input.numel(); ++i)
    std::memcpy(result.data() + i * input.element_size(),
                pointer(view, offset(view, i)), input.element_size());
  return result;
}

/** Snapshot a broadcast mask before a potentially aliased mutation. */
std::vector<std::uint8_t> flags(const DenseTensor &mask,
                                std::span<const std::int64_t> shape,
                                std::int64_t count) {
  cpu(mask);
  if (mask.scalar_type() == kFloat8E4M3FN)
    throw std::invalid_argument("CPU FP8 mask is format-specific");
  const auto view = align_for_broadcast(mask, shape);
  std::vector<std::uint8_t> result(static_cast<std::size_t>(count));
  for (std::int64_t i = 0; i < count; ++i)
    result[static_cast<std::size_t>(i)] =
        load<double>(pointer(view, offset(view, i)), mask.scalar_type()) != 0;
  return result;
}
} // namespace

DenseTensor index_select(const DenseTensor &input, std::int64_t dimension,
                         const DenseTensor &index) {
  cpu(input);
  same_device(input, index);
  const auto axis = normalize_dimension(dimension, input.dim());
  const auto selected = indices(index, input.size(axis), true);
  auto shape = input.sizes().vec();
  shape[axis] = index.numel();
  geometry(shape, input.element_size());
  auto output = empty(shape, input.options());
  const auto in = input.view_descriptor();
  const auto out = output.view_descriptor();
  for (std::int64_t i = 0; i < output.numel(); ++i) {
    auto coord = coordinates(i, shape);
    coord[axis] = selected[static_cast<std::size_t>(coord[axis])];
    std::memcpy(pointer(out, i), pointer(in, at(in, coord)),
                input.element_size());
  }
  return output;
}

DenseTensor gather(const DenseTensor &input, std::int64_t dimension,
                   const DenseTensor &index) {
  cpu(input);
  same_device(input, index);
  if (input.dim() != index.dim())
    throw std::invalid_argument("gather rank mismatch");
  const auto axis = normalize_dimension(dimension, input.dim());
  for (std::int64_t d = 0; d < input.dim(); ++d)
    if (static_cast<std::size_t>(d) != axis && index.size(d) > input.size(d))
      throw std::invalid_argument("gather shape exceeds input");
  const auto selected = indices(index, input.size(axis), false);
  auto shape = index.sizes().vec();
  geometry(shape, input.element_size());
  auto output = empty(shape, input.options());
  const auto in = input.view_descriptor();
  const auto out = output.view_descriptor();
  for (std::int64_t i = 0; i < output.numel(); ++i) {
    auto coord = coordinates(i, shape);
    coord[axis] = selected[static_cast<std::size_t>(i)];
    std::memcpy(pointer(out, i), pointer(in, at(in, coord)),
                input.element_size());
  }
  return output;
}

DenseTensor repeat(const DenseTensor &input,
                   std::span<const std::int64_t> repeats) {
  cpu(input);
  if (repeats.size() < static_cast<std::size_t>(input.dim()) ||
      repeats.size() > kMaximumTensorRank)
    throw std::invalid_argument("repeat rank mismatch");
  const auto pad = repeats.size() - static_cast<std::size_t>(input.dim());
  std::vector<std::int64_t> shape(repeats.size());
  std::int64_t count = 1;
  for (std::size_t d = 0; d < repeats.size(); ++d) {
    shape[d] = product(d < pad ? 1 : input.size(d - pad), repeats[d]);
    count = product(count, shape[d]);
  }
  product(count, static_cast<std::int64_t>(input.element_size()));
  auto output = empty(shape, input.options());
  const auto in = input.view_descriptor();
  const auto out = output.view_descriptor();
  for (std::int64_t i = 0; i < output.numel(); ++i) {
    auto coord = coordinates(i, shape);
    std::int64_t input_offset = 0;
    for (std::size_t d = pad; d < shape.size(); ++d)
      input_offset += (coord[d] % input.size(d - pad)) * in.strides[d - pad];
    std::memcpy(pointer(out, i), pointer(in, input_offset),
                input.element_size());
  }
  return output;
}

DenseTensor repeat_interleave(const DenseTensor &input, std::int64_t repeats,
                              std::int64_t dimension) {
  cpu(input);
  const auto axis = normalize_dimension(dimension, input.dim());
  auto shape = input.sizes().vec();
  shape[axis] = product(shape[axis], repeats);
  std::int64_t count = 1;
  for (auto size : shape)
    count = product(count, size);
  product(count, static_cast<std::int64_t>(input.element_size()));
  auto output = empty(shape, input.options());
  const auto in = input.view_descriptor();
  const auto out = output.view_descriptor();
  for (std::int64_t i = 0; i < output.numel(); ++i) {
    auto coord = coordinates(i, shape);
    coord[axis] /= repeats;
    std::memcpy(pointer(out, i), pointer(in, at(in, coord)),
                input.element_size());
  }
  return output;
}

void index_copy(DenseTensor &destination, std::int64_t dimension,
                const DenseTensor &index, const DenseTensor &source) {
  cpu(destination);
  same_device(destination, source);
  same_device(destination, index);
  if (destination.dim() != source.dim() ||
      destination.scalar_type() != source.scalar_type())
    throw std::invalid_argument("index_copy rank or dtype mismatch");
  const auto axis = normalize_dimension(dimension, destination.dim());
  const auto selected = indices(index, destination.size(axis), true);
  for (std::int64_t d = 0; d < destination.dim(); ++d) {
    const auto expected = static_cast<std::size_t>(d) == axis
                              ? index.numel()
                              : destination.size(d);
    if (source.size(d) != expected)
      throw std::invalid_argument("index_copy shape mismatch");
  }
  const auto values = snapshot(source);
  const auto shape = source.sizes().vec();
  const auto out = destination.view_descriptor();
  for (std::int64_t i = 0; i < source.numel(); ++i) {
    auto coord = coordinates(i, shape);
    coord[axis] = selected[static_cast<std::size_t>(coord[axis])];
    std::memcpy(pointer(out, at(out, coord)),
                values.data() + i * source.element_size(),
                source.element_size());
  }
}

void index_fill(DenseTensor &destination, std::int64_t dimension,
                const DenseTensor &index, double value) {
  cpu(destination);
  same_device(destination, index);
  const auto axis = normalize_dimension(dimension, destination.dim());
  const auto selected = indices(index, destination.size(axis), true);
  auto shape = destination.sizes().vec();
  shape[axis] = index.numel();
  std::int64_t count = 1;
  for (auto size : shape)
    count = product(count, size);
  const auto out = destination.view_descriptor();
  alignas(std::max_align_t) std::array<std::byte, sizeof(double)> encoded{};
  store(encoded.data(), destination.scalar_type(), value);
  for (std::int64_t i = 0; i < count; ++i) {
    auto coord = coordinates(i, shape);
    coord[axis] = selected[static_cast<std::size_t>(coord[axis])];
    std::memcpy(pointer(out, at(out, coord)), encoded.data(),
                destination.element_size());
  }
}

void masked_fill(DenseTensor &destination, const DenseTensor &mask,
                 double value) {
  cpu(destination);
  same_device(destination, mask);
  const auto shape = destination.sizes().vec();
  const auto selected = flags(mask, shape, destination.numel());
  const auto out = destination.view_descriptor();
  alignas(std::max_align_t) std::array<std::byte, sizeof(double)> encoded{};
  store(encoded.data(), destination.scalar_type(), value);
  for (std::int64_t i = 0; i < destination.numel(); ++i)
    if (selected[static_cast<std::size_t>(i)])
      std::memcpy(pointer(out, offset(out, i)), encoded.data(),
                  destination.element_size());
}

DenseTensor masked_select(const DenseTensor &input, const DenseTensor &mask) {
  cpu(input);
  same_device(input, mask);
  const auto shape = input.sizes().vec();
  const auto selected = flags(mask, shape, input.numel());
  std::int64_t count = 0;
  for (const auto flag : selected)
    count += flag != 0;
  product(count, static_cast<std::int64_t>(input.element_size()));
  auto output = empty({count}, input.options());
  const auto in = input.view_descriptor();
  const auto out = output.view_descriptor();
  std::int64_t next = 0;
  for (std::int64_t i = 0; i < input.numel(); ++i)
    if (selected[static_cast<std::size_t>(i)])
      std::memcpy(pointer(out, next++), pointer(in, offset(in, i)),
                  input.element_size());
  return output;
}
} // namespace tyloi::cuda::host
