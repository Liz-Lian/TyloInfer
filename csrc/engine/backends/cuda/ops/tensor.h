#pragma once

#include "engine/tensor.h"

namespace tyloi::cuda::ops {

void decode_commit(tyloi::DenseTensor next, tyloi::DenseTensor generated, tyloi::DenseTensor step,
                   tyloi::DenseTensor input, tyloi::DenseTensor position, tyloi::DenseTensor length);
tyloi::DenseTensor tensor(std::span<const float> values, Device target);
tyloi::DenseTensor tensor(std::span<const std::int32_t> values, Device target);
tyloi::DenseTensor tensor(std::span<const std::int64_t> values, Device target);
tyloi::DenseTensor tensor(std::span<const std::uint8_t> values, Device target);
tyloi::DenseTensor from_blob(void* data, std::span<const std::int64_t> shape, const TensorOptions& options);
tyloi::DenseTensor gelu(tyloi::DenseTensor input, bool approximate);
tyloi::DenseTensor silu(tyloi::DenseTensor input);
tyloi::DenseTensor relu_squared(tyloi::DenseTensor input);
tyloi::DenseTensor softcap(tyloi::DenseTensor input, double cap);
tyloi::DenseTensor scale(tyloi::DenseTensor input, double factor);
tyloi::DenseTensor divide(tyloi::DenseTensor input, double divisor);
tyloi::DenseTensor copy(tyloi::DenseTensor input, Device target);
void copy_into(tyloi::DenseTensor destination, tyloi::DenseTensor source);
tyloi::DenseTensor cast(tyloi::DenseTensor input, ScalarType type);
tyloi::DenseTensor cast(tyloi::DenseTensor input, ScalarType type, Device target);
tyloi::DenseTensor contiguous(tyloi::DenseTensor input);
tyloi::DenseTensor clone(tyloi::DenseTensor input);
tyloi::DenseTensor cat(std::span<const tyloi::DenseTensor> inputs, std::int64_t dim);
tyloi::DenseTensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options);
tyloi::DenseTensor index_select(tyloi::DenseTensor input, std::int64_t dim, tyloi::DenseTensor indices);
tyloi::DenseTensor sum(tyloi::DenseTensor input, std::int64_t dim, bool keepdim);
void index_copy(tyloi::DenseTensor input, std::int64_t dim, tyloi::DenseTensor indices, tyloi::DenseTensor source);
tyloi::DenseTensor arange(std::int64_t first, std::int64_t last, Device device);
tyloi::DenseTensor full(std::span<const std::int64_t> shape, std::int64_t value, Device device);
tyloi::DenseTensor add(tyloi::DenseTensor input, std::int64_t offset);
tyloi::DenseTensor add(tyloi::DenseTensor left, tyloi::DenseTensor right);
tyloi::DenseTensor multiply(tyloi::DenseTensor left, tyloi::DenseTensor right);
tyloi::DenseTensor matmul(tyloi::DenseTensor left, tyloi::DenseTensor right);
tyloi::DenseTensor add_residual(tyloi::DenseTensor residual, tyloi::DenseTensor value, bool configured);
// Numeric conveniences shared with DenseTensor members; thin dispatch to native ops.
tyloi::DenseTensor mean(tyloi::DenseTensor input, std::int64_t dim, bool keepdim);
tyloi::DenseTensor amax(tyloi::DenseTensor input, std::int64_t dim, bool keepdim);
tyloi::DenseTensor argmax(tyloi::DenseTensor input, std::int64_t dim, bool keepdim);
tyloi::DenseTensor all(tyloi::DenseTensor input, std::int64_t dim, bool keepdim);
tyloi::DenseTensor any(tyloi::DenseTensor input, std::int64_t dim, bool keepdim);
tyloi::DenseTensor abs(tyloi::DenseTensor input);
tyloi::DenseTensor square(tyloi::DenseTensor input);
tyloi::DenseTensor exp(tyloi::DenseTensor input);
tyloi::DenseTensor sqrt(tyloi::DenseTensor input);
tyloi::DenseTensor logical_not(tyloi::DenseTensor input);
tyloi::DenseTensor clamp(tyloi::DenseTensor input, double minimum, double maximum);
tyloi::DenseTensor clamp_min(tyloi::DenseTensor input, double minimum);
tyloi::DenseTensor clamp_max(tyloi::DenseTensor input, double maximum);
tyloi::DenseTensor remainder(tyloi::DenseTensor input, double divisor);
tyloi::DenseTensor eq(tyloi::DenseTensor left, tyloi::DenseTensor right);
tyloi::DenseTensor eq(tyloi::DenseTensor input, double other);
tyloi::DenseTensor ne(tyloi::DenseTensor left, tyloi::DenseTensor right);
tyloi::DenseTensor ne(tyloi::DenseTensor input, double other);
bool equal(const tyloi::DenseTensor& left, const tyloi::DenseTensor& right);
tyloi::DenseTensor masked_select(tyloi::DenseTensor input, tyloi::DenseTensor mask);
tyloi::DenseTensor gather(tyloi::DenseTensor input, std::int64_t dim, tyloi::DenseTensor indices);
tyloi::DenseTensor repeat(tyloi::DenseTensor input, std::span<const std::int64_t> repeats);
tyloi::DenseTensor repeat_interleave(tyloi::DenseTensor input, std::int64_t repeats, std::int64_t dim);
tyloi::DenseTensor tril(tyloi::DenseTensor input, std::int64_t diagonal);
tyloi::DenseTensor masked_fill(tyloi::DenseTensor input, tyloi::DenseTensor mask, double value);
void masked_fill_inplace(tyloi::DenseTensor input, tyloi::DenseTensor mask, double value);
// CPU and CUDA paths; CUDA requires an execution scope.
void fill(tyloi::DenseTensor input, double value);
void scatter_add(tyloi::DenseTensor input, std::int64_t dim, tyloi::DenseTensor indices, tyloi::DenseTensor source);
void scatter(tyloi::DenseTensor input, std::int64_t dim, tyloi::DenseTensor indices, tyloi::DenseTensor source);
void index_fill(tyloi::DenseTensor input, std::int64_t dim, tyloi::DenseTensor indices, double value);

} // namespace tyloi::cuda::ops
