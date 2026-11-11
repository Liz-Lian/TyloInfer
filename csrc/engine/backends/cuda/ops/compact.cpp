#include "engine/ops/compact.h"

#ifdef TYLOINFER_NATIVE_CUDA
#include "backend.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"
#include "engine/backends/cuda/native/memory.h"
#include "engine/backends/cuda/native/tensor/backend.h"

#include <iostream>
#include <stdexcept>
#include <utility>

namespace tyloi::ops {
struct Compact::Impl {
    cuda::HostBuffer host{128};
    DenseTensor storage = cuda::zeros({1}, TensorOptions().device(kCUDA).dtype(kInt32));
    int previous = -1;
    Impl() { cuda::charge_tensor_host_bytes(host.size()); }
    ~Impl() { cuda::tensor_host_bytes.fetch_sub(host.size()); }
};

Compact::Compact() {
    (void)cuda::Executor::current();
    impl_ = std::make_unique<Impl>();
}
Compact::~Compact() = default;
Compact::Compact(Compact&&) noexcept = default;
Compact& Compact::operator=(Compact&&) noexcept = default;

void Compact::set(bool enabled, const char* role, std::int64_t rows) {
    (void)cuda::Executor::current();
    if (impl_->previous == int(enabled)) return;
    const MfqCudaGuard guard(impl_->storage.device());
    *static_cast<std::int32_t*>(impl_->host.data()) = enabled ? 1 : 0;
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(impl_->storage.data_ptr(), impl_->host.data(), sizeof(std::int32_t),
        cudaMemcpyHostToDevice, mfq_current_cuda_stream()));
    std::cerr << "runtime_mtp_compact_control role=" << role << " rows=" << rows << " enabled=" << int(enabled) << '\n';
    impl_->previous = int(enabled);
}
Tensor Compact::forward(const Tensor& logits) const {
    (void)cuda::Executor::current();
    const auto* value = logits.representation<DenseTensor>();
    if (!value || !value->defined() || !value->is_cuda() || value->device() != impl_->storage.device())
        throw std::invalid_argument("compact logits must be dense on the control device");
    Tensor result{DenseTensor{}};
    result.dense() = sample_top_k_compact_if_cuda(*value, 64, impl_->storage.data_ptr<std::int32_t>());
    return result;
}
} // namespace tyloi::ops
#endif
