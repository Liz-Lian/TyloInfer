#include "engine/backends/cuda/ops/tensor.h"

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_activation_ops.h"

#include <optional>
#include <vector>

namespace mfq::cuda::ops {
namespace tb = mfq_tensor_backend;

template <typename Value>
static mfq::Tensor tensor_values(std::span<const Value> values, Device target) {
    return tb::tensor(values, TensorOptions().dtype(tb::scalar_type_for<Value>()).device(target));
}
mfq::Tensor tensor(std::span<const float> values, Device target) { return tensor_values(values, target); }
mfq::Tensor tensor(std::span<const std::int32_t> values, Device target) { return tensor_values(values, target); }
mfq::Tensor tensor(std::span<const std::int64_t> values, Device target) { return tensor_values(values, target); }
mfq::Tensor tensor(std::span<const std::uint8_t> values, Device target) { return tensor_values(values, target); }
mfq::Tensor from_blob(void* data, std::span<const std::int64_t> shape, const TensorOptions& options) {
    return tb::from_blob(data, shape, options);
}
mfq::Tensor gelu(mfq::Tensor input, bool approximate) {
    MfqCudaGuard guard(input.device().index);
    return tb::gelu(tb::Tensor(input), approximate ? "tanh" : "none");
}
mfq::Tensor silu(mfq::Tensor input) {
    MfqCudaGuard guard(input.device().index);
    return tb::silu(tb::Tensor(input));
}
mfq::Tensor relu_squared(mfq::Tensor input) {
    MfqCudaGuard guard(input.device().index);
    return relu_squared_cuda(tb::Tensor(input).contiguous());
}
mfq::Tensor softcap(mfq::Tensor input, double cap) {
    if (!(cap > 0.0)) return input;
    MfqCudaGuard guard(input.device().index);
    return tb::tanh(tb::Tensor(input) / cap) * cap;
}
mfq::Tensor scale(mfq::Tensor input, double factor) {
    MfqCudaGuard guard(input.device().index);
    return tb::Tensor(input) * factor;
}
mfq::Tensor copy(mfq::Tensor input, Device target) {
    return target.is_cuda() ? Executor::current().collectives().copy(tb::Tensor(input), target.index)
                            : tb::Tensor(input).to(target);
}
void copy_into(mfq::Tensor destination, mfq::Tensor source) {
    std::optional<MfqCudaGuard> guard;
    if (destination.device().is_cuda()) guard.emplace(destination.device().index);
    else if (source.device().is_cuda()) guard.emplace(source.device().index);
    tb::Tensor(destination).copy_(tb::Tensor(source));
}
mfq::Tensor cast(mfq::Tensor input, ScalarType type) { return tb::Tensor(input).to(type); }
mfq::Tensor cast(mfq::Tensor input, ScalarType type, Device target) { return tb::Tensor(input).to(target, type); }
mfq::Tensor contiguous(mfq::Tensor input) { return tb::Tensor(input).contiguous(); }
mfq::Tensor clone(mfq::Tensor input) { return tb::Tensor(input).clone(); }
mfq::Tensor reshape(mfq::Tensor input, std::span<const std::int64_t> shape) {
    return tb::Tensor(input).reshape(shape);
}
mfq::Tensor expand(mfq::Tensor input, std::span<const std::int64_t> shape) {
    return tb::Tensor(input).expand(shape);
}
mfq::Tensor cat(std::span<const mfq::Tensor> inputs, std::int64_t dim) {
    std::optional<MfqCudaGuard> guard;
    if (inputs.front().device().is_cuda()) guard.emplace(inputs.front().device().index);
    std::vector<tb::Tensor> tensors;
    tensors.reserve(inputs.size());
    for (const auto& input : inputs) tensors.emplace_back(input);
    return tb::cat(tensors, dim);
}
mfq::Tensor zeros(std::span<const std::int64_t> shape, const TensorOptions& options) {
    std::optional<MfqCudaGuard> guard;
    if (options.target_device().is_cuda()) guard.emplace(options.target_device().index);
    return tb::zeros(shape, options);
}
mfq::Tensor select(mfq::Tensor input, std::int64_t dim, std::int64_t index) {
    return tb::Tensor(input).select(dim, index);
}
mfq::Tensor unsqueeze(mfq::Tensor input, std::int64_t dim) {
    return tb::Tensor(input).unsqueeze(dim);
}
mfq::Tensor slice(mfq::Tensor input, std::int64_t dim, std::int64_t start, std::int64_t end, std::int64_t step) {
    return tb::Tensor(input).slice(dim, start, end, step);
}
mfq::Tensor index_select(mfq::Tensor input, std::int64_t dim, mfq::Tensor indices) {
    MfqCudaGuard guard(input.device().index);
    return tb::Tensor(input).index_select(dim, tb::Tensor(indices));
}
mfq::Tensor sum(mfq::Tensor input, std::int64_t dim, bool keepdim) {
    MfqCudaGuard guard(input.device().index);
    return tb::sum(tb::Tensor(input), dim, keepdim);
}
void index_copy(mfq::Tensor input, std::int64_t dim, mfq::Tensor indices, mfq::Tensor source) {
    MfqCudaGuard guard(input.device().index);
    tb::Tensor(input).index_copy_(dim, tb::Tensor(indices), tb::Tensor(source));
}
std::vector<mfq::Tensor> chunk(mfq::Tensor input, std::int64_t count, std::int64_t dim) {
    auto parts = tb::Tensor(input).chunk(count, dim);
    return {parts.begin(), parts.end()};
}
mfq::Tensor arange(std::int64_t first, std::int64_t last, Device device) {
    std::optional<MfqCudaGuard> guard;
    if (device.is_cuda()) guard.emplace(device.index);
    return tb::arange(first, last, tb::TensorOptions().device(device).dtype(tb::kInt64));
}
mfq::Tensor full(std::span<const std::int64_t> shape, std::int64_t value, Device device) {
    std::optional<MfqCudaGuard> guard;
    if (device.is_cuda()) guard.emplace(device.index);
    return tb::full(shape, value, tb::TensorOptions().device(device).dtype(tb::kInt64));
}
mfq::Tensor add(mfq::Tensor input, std::int64_t offset) {
    MfqCudaGuard guard(input.device().index);
    return tb::Tensor(input) + offset;
}
mfq::Tensor add(mfq::Tensor left, mfq::Tensor right) {
    MfqCudaGuard guard(left.device().index);
    return tb::Tensor(left) + tb::Tensor(right);
}
mfq::Tensor multiply(mfq::Tensor left, mfq::Tensor right) {
    MfqCudaGuard guard(left.device().index);
    return tb::Tensor(left) * tb::Tensor(right);
}
mfq::Tensor matmul(mfq::Tensor left, mfq::Tensor right) {
    MfqCudaGuard guard(left.device().index);
    return tb::matmul(tb::Tensor(left), tb::Tensor(right));
}
mfq::Tensor transpose(mfq::Tensor input, std::int64_t first, std::int64_t second) {
    return tb::Tensor(input).transpose(first, second);
}
mfq::Tensor add_residual(mfq::Tensor residual, mfq::Tensor value) {
    MfqCudaGuard guard(residual.device().index);
    return acc_cuda(tb::Tensor(residual), tb::Tensor(value));
}

} // namespace mfq::cuda::ops
#endif
