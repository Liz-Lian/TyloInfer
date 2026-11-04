#include "engine/backends/cuda/ops/tensor.h"

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_activation_ops.h"

#include <optional>
#include <vector>

namespace tyloi::cuda::ops {

template <typename Value>
static tyloi::Tensor tensor_values(std::span<const Value> values, Device target) {
    return ::tyloi::cuda::tensor(values, TensorOptions().dtype(::tyloi::cuda::scalar_type_for<Value>()).device(target));
}
tyloi::Tensor tensor(std::span<const float> values, Device target) { return tensor_values(values, target); }
tyloi::Tensor tensor(std::span<const std::int32_t> values, Device target) { return tensor_values(values, target); }
tyloi::Tensor tensor(std::span<const std::int64_t> values, Device target) { return tensor_values(values, target); }
tyloi::Tensor tensor(std::span<const std::uint8_t> values, Device target) { return tensor_values(values, target); }
tyloi::Tensor from_blob(void* data, std::span<const std::int64_t> shape, const TensorOptions& options) {
    return ::tyloi::cuda::from_blob(data, shape, options);
}
tyloi::Tensor gelu(tyloi::Tensor input, bool approximate) {
    MfqCudaGuard guard(input.device().index);
    return ::tyloi::cuda::gelu(input, approximate ? "tanh" : "none");
}
tyloi::Tensor silu(tyloi::Tensor input) {
    MfqCudaGuard guard(input.device().index);
    return ::tyloi::cuda::silu(input);
}
tyloi::Tensor relu_squared(tyloi::Tensor input) {
    MfqCudaGuard guard(input.device().index);
    return relu_squared_cuda(::tyloi::cuda::contiguous(input));
}
tyloi::Tensor softcap(tyloi::Tensor input, double cap) {
    if (!(cap > 0.0)) return input;
    MfqCudaGuard guard(input.device().index);
    auto scaled = ::tyloi::cuda::scalar_binary_cuda(input, cap, 3);
    return ::tyloi::cuda::scalar_binary_cuda(::tyloi::cuda::tanh(scaled), cap, 2);
}
tyloi::Tensor scale(tyloi::Tensor input, double factor) {
    MfqCudaGuard guard(input.device().index);
    return ::tyloi::cuda::scalar_binary_cuda(input, factor, 2);
}
tyloi::Tensor copy(tyloi::Tensor input, Device target) {
    return target.is_cuda() ? Executor::current().collectives().copy(input, target.index)
                            : ::tyloi::cuda::copy(input, target);
}
void copy_into(tyloi::Tensor destination, tyloi::Tensor source) {
    std::optional<MfqCudaGuard> guard;
    if (destination.device().is_cuda()) guard.emplace(destination.device().index);
    else if (source.device().is_cuda()) guard.emplace(source.device().index);
    ::tyloi::cuda::copy_into(destination, source);
}
tyloi::Tensor cast(tyloi::Tensor input, ScalarType type) { return ::tyloi::cuda::cast(input, type); }
tyloi::Tensor cast(tyloi::Tensor input, ScalarType type, Device target) { return ::tyloi::cuda::cast(input, type, target); }
tyloi::Tensor contiguous(tyloi::Tensor input) { return ::tyloi::cuda::contiguous(input); }
tyloi::Tensor clone(tyloi::Tensor input) { return ::tyloi::cuda::clone(input); }
tyloi::Tensor reshape(tyloi::Tensor input, std::span<const std::int64_t> shape) {
    return ::tyloi::cuda::reshape(input, shape);
}
tyloi::Tensor expand(tyloi::Tensor input, std::span<const std::int64_t> shape) {
    return ::tyloi::cuda::expand(input, shape);
}
tyloi::Tensor cat(std::span<const tyloi::Tensor> inputs, std::int64_t dim) {
    std::optional<MfqCudaGuard> guard;
    if (inputs.front().device().is_cuda()) guard.emplace(inputs.front().device().index);
    return ::tyloi::cuda::cat(inputs, dim);
}
tyloi::Tensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options) {
    std::optional<MfqCudaGuard> guard;
    if (options.target_device().is_cuda()) guard.emplace(options.target_device().index);
    return ::tyloi::cuda::zeros(shape, options);
}
tyloi::Tensor select(tyloi::Tensor input, std::int64_t dim, std::int64_t index) {
    return ::tyloi::cuda::select(input, dim, index);
}
tyloi::Tensor unsqueeze(tyloi::Tensor input, std::int64_t dim) {
    return ::tyloi::cuda::unsqueeze(input, dim);
}
tyloi::Tensor slice(tyloi::Tensor input, std::int64_t dim, std::int64_t start, std::int64_t end, std::int64_t step) {
    return ::tyloi::cuda::slice(input, dim, start, end, step);
}
tyloi::Tensor index_select(tyloi::Tensor input, std::int64_t dim, tyloi::Tensor indices) {
    MfqCudaGuard guard(input.device().index);
    return ::tyloi::cuda::index_select_cuda(input, dim, indices);
}
tyloi::Tensor sum(tyloi::Tensor input, std::int64_t dim, bool keepdim) {
    MfqCudaGuard guard(input.device().index);
    return ::tyloi::cuda::sum(input, dim, keepdim);
}
void index_copy(tyloi::Tensor input, std::int64_t dim, tyloi::Tensor indices, tyloi::Tensor source) {
    MfqCudaGuard guard(input.device().index);
    ::tyloi::cuda::index_copy_cuda(input, dim, indices, source);
}
std::vector<tyloi::Tensor> chunk(tyloi::Tensor input, std::int64_t count, std::int64_t dim) {
    return ::tyloi::cuda::chunk(input, count, dim);
}
tyloi::Tensor arange(std::int64_t first, std::int64_t last, Device device) {
    std::optional<MfqCudaGuard> guard;
    if (device.is_cuda()) guard.emplace(device.index);
    return ::tyloi::cuda::arange(first, last, ::tyloi::TensorOptions().device(device).dtype(::tyloi::kInt64));
}
tyloi::Tensor full(std::span<const std::int64_t> shape, std::int64_t value, Device device) {
    std::optional<MfqCudaGuard> guard;
    if (device.is_cuda()) guard.emplace(device.index);
    return ::tyloi::cuda::full(shape, value, ::tyloi::TensorOptions().device(device).dtype(::tyloi::kInt64));
}
tyloi::Tensor add(tyloi::Tensor input, std::int64_t offset) {
    MfqCudaGuard guard(input.device().index);
    return ::tyloi::cuda::scalar_binary_cuda(input, static_cast<double>(offset), 0);
}
tyloi::Tensor add(tyloi::Tensor left, tyloi::Tensor right) {
    MfqCudaGuard guard(left.device().index);
    return ::tyloi::cuda::binary_cuda(left, right, 0);
}
tyloi::Tensor multiply(tyloi::Tensor left, tyloi::Tensor right) {
    MfqCudaGuard guard(left.device().index);
    return ::tyloi::cuda::binary_cuda(left, right, 2);
}
tyloi::Tensor matmul(tyloi::Tensor left, tyloi::Tensor right) {
    MfqCudaGuard guard(left.device().index);
    return ::tyloi::cuda::matmul(left, right);
}
tyloi::Tensor transpose(tyloi::Tensor input, std::int64_t first, std::int64_t second) {
    return ::tyloi::cuda::transpose(input, first, second);
}
tyloi::Tensor add_residual(tyloi::Tensor residual, tyloi::Tensor value) {
    MfqCudaGuard guard(residual.device().index);
    return acc_cuda(residual, value);
}

} // namespace tyloi::cuda::ops
#endif
