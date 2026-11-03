#pragma once

#include "engine/tensor.h"

namespace mfq::cuda::ops {

mfq::Tensor tensor(std::span<const float> values, Device target);
mfq::Tensor tensor(std::span<const std::int32_t> values, Device target);
mfq::Tensor tensor(std::span<const std::int64_t> values, Device target);
mfq::Tensor tensor(std::span<const std::uint8_t> values, Device target);
mfq::Tensor from_blob(void* data, std::span<const std::int64_t> shape, const TensorOptions& options);
mfq::Tensor gelu(mfq::Tensor input, bool approximate);
mfq::Tensor silu(mfq::Tensor input);
mfq::Tensor relu_squared(mfq::Tensor input);
mfq::Tensor softcap(mfq::Tensor input, double cap);
mfq::Tensor scale(mfq::Tensor input, double factor);
mfq::Tensor copy(mfq::Tensor input, Device target);
void copy_into(mfq::Tensor destination, mfq::Tensor source);
mfq::Tensor cast(mfq::Tensor input, ScalarType type);
mfq::Tensor cast(mfq::Tensor input, ScalarType type, Device target);
mfq::Tensor contiguous(mfq::Tensor input);
mfq::Tensor clone(mfq::Tensor input);
mfq::Tensor reshape(mfq::Tensor input, std::span<const std::int64_t> shape);
mfq::Tensor expand(mfq::Tensor input, std::span<const std::int64_t> shape);
mfq::Tensor cat(std::span<const mfq::Tensor> inputs, std::int64_t dim);
mfq::Tensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options);
mfq::Tensor select(mfq::Tensor input, std::int64_t dim, std::int64_t index);
mfq::Tensor unsqueeze(mfq::Tensor input, std::int64_t dim);
mfq::Tensor slice(mfq::Tensor input, std::int64_t dim, std::int64_t start, std::int64_t end, std::int64_t step = 1);
mfq::Tensor index_select(mfq::Tensor input, std::int64_t dim, mfq::Tensor indices);
mfq::Tensor sum(mfq::Tensor input, std::int64_t dim, bool keepdim);
void index_copy(mfq::Tensor input, std::int64_t dim, mfq::Tensor indices, mfq::Tensor source);
std::vector<mfq::Tensor> chunk(mfq::Tensor input, std::int64_t count, std::int64_t dim);
mfq::Tensor arange(std::int64_t first, std::int64_t last, Device device);
mfq::Tensor full(std::span<const std::int64_t> shape, std::int64_t value, Device device);
mfq::Tensor add(mfq::Tensor input, std::int64_t offset);
mfq::Tensor add(mfq::Tensor left, mfq::Tensor right);
mfq::Tensor multiply(mfq::Tensor left, mfq::Tensor right);
mfq::Tensor matmul(mfq::Tensor left, mfq::Tensor right);
mfq::Tensor transpose(mfq::Tensor input, std::int64_t first, std::int64_t second);
mfq::Tensor add_residual(mfq::Tensor residual, mfq::Tensor value);

} // namespace mfq::cuda::ops
