#pragma once

#include "engine/tensor.h"

namespace tyloi::cuda::ops {

tyloi::Tensor tensor(std::span<const float> values, Device target);
tyloi::Tensor tensor(std::span<const std::int32_t> values, Device target);
tyloi::Tensor tensor(std::span<const std::int64_t> values, Device target);
tyloi::Tensor tensor(std::span<const std::uint8_t> values, Device target);
tyloi::Tensor from_blob(void* data, std::span<const std::int64_t> shape, const TensorOptions& options);
tyloi::Tensor gelu(tyloi::Tensor input, bool approximate);
tyloi::Tensor silu(tyloi::Tensor input);
tyloi::Tensor relu_squared(tyloi::Tensor input);
tyloi::Tensor softcap(tyloi::Tensor input, double cap);
tyloi::Tensor scale(tyloi::Tensor input, double factor);
tyloi::Tensor copy(tyloi::Tensor input, Device target);
void copy_into(tyloi::Tensor destination, tyloi::Tensor source);
tyloi::Tensor cast(tyloi::Tensor input, ScalarType type);
tyloi::Tensor cast(tyloi::Tensor input, ScalarType type, Device target);
tyloi::Tensor contiguous(tyloi::Tensor input);
tyloi::Tensor clone(tyloi::Tensor input);
tyloi::Tensor reshape(tyloi::Tensor input, std::span<const std::int64_t> shape);
tyloi::Tensor expand(tyloi::Tensor input, std::span<const std::int64_t> shape);
tyloi::Tensor cat(std::span<const tyloi::Tensor> inputs, std::int64_t dim);
tyloi::Tensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options);
tyloi::Tensor select(tyloi::Tensor input, std::int64_t dim, std::int64_t index);
tyloi::Tensor unsqueeze(tyloi::Tensor input, std::int64_t dim);
tyloi::Tensor slice(tyloi::Tensor input, std::int64_t dim, std::int64_t start, std::int64_t end, std::int64_t step = 1);
tyloi::Tensor index_select(tyloi::Tensor input, std::int64_t dim, tyloi::Tensor indices);
tyloi::Tensor sum(tyloi::Tensor input, std::int64_t dim, bool keepdim);
void index_copy(tyloi::Tensor input, std::int64_t dim, tyloi::Tensor indices, tyloi::Tensor source);
std::vector<tyloi::Tensor> chunk(tyloi::Tensor input, std::int64_t count, std::int64_t dim);
tyloi::Tensor arange(std::int64_t first, std::int64_t last, Device device);
tyloi::Tensor full(std::span<const std::int64_t> shape, std::int64_t value, Device device);
tyloi::Tensor add(tyloi::Tensor input, std::int64_t offset);
tyloi::Tensor add(tyloi::Tensor left, tyloi::Tensor right);
tyloi::Tensor multiply(tyloi::Tensor left, tyloi::Tensor right);
tyloi::Tensor matmul(tyloi::Tensor left, tyloi::Tensor right);
tyloi::Tensor transpose(tyloi::Tensor input, std::int64_t first, std::int64_t second);
tyloi::Tensor add_residual(tyloi::Tensor residual, tyloi::Tensor value);

} // namespace tyloi::cuda::ops
