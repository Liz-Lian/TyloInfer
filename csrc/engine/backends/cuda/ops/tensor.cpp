#include "engine/backends/cuda/ops/tensor.h"

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"

#include <optional>

namespace mfq::cuda::ops {
namespace tb = mfq_tensor_backend;

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
mfq::Tensor cast(mfq::Tensor input, ScalarType type) { return tb::Tensor(input).to(type); }
mfq::Tensor cast(mfq::Tensor input, ScalarType type, Device target) { return tb::Tensor(input).to(target, type); }
mfq::Tensor contiguous(mfq::Tensor input) { return tb::Tensor(input).contiguous(); }
mfq::Tensor clone(mfq::Tensor input) { return tb::Tensor(input).clone(); }
mfq::Tensor reshape(mfq::Tensor input, std::span<const std::int64_t> shape) {
    return tb::Tensor(input).reshape(shape);
}
mfq::Tensor slice(mfq::Tensor input, std::int64_t dim, std::int64_t start, std::int64_t end, std::int64_t step) {
    return tb::Tensor(input).slice(dim, start, end, step);
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
