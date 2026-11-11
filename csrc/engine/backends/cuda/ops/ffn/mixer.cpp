#include "engine/ops/residual.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/ops/ffn/mixer_core.h"
#include <stdexcept>
#include <utility>

namespace tyloi::ops {
struct Residual::Impl {
    cuda::ResidualCore core;
    Impl(ResidualWeights weights, ResidualConfig config) : core(std::move(weights),config) {}
};
Residual::Residual() = default;
Residual::Residual(ResidualWeights weights, ResidualConfig config)
    : impl_(std::make_unique<Impl>(std::move(weights),config)) {}
Residual::~Residual() = default;
Residual::Residual(Residual&&) noexcept = default;
Residual& Residual::operator=(Residual&&) noexcept = default;
Residual::Result Residual::pre(const Tensor& input, ResidualRole role) const {
    Result output{Tensor{DenseTensor{}},Tensor{DenseTensor{}},Tensor{DenseTensor{}}};
    if (!impl_) throw std::logic_error("gated-residual operator is not initialized");
    const MfqCudaGuard guard(input.device());
    auto result=impl_->core.pre(input.dense(),role);
    output.mixed.dense()=std::move(result.mixed);
    output.residual.dense()=std::move(result.residual);
    output.gate.dense()=std::move(result.gate);
    return output;
}
Residual::Result Residual::pre_after(const Tensor& input, const Tensor& residual, const Tensor& gate,
                                     ResidualRole role) const {
    Result output{Tensor{DenseTensor{}},Tensor{DenseTensor{}},Tensor{DenseTensor{}}};
    if (!impl_) throw std::logic_error("gated-residual operator is not initialized");
    const MfqCudaGuard guard(residual.device());
    auto result=impl_->core.pre_after(input.dense(),residual.dense(),gate.dense(),role);
    output.mixed.dense()=std::move(result.mixed);
    output.residual.dense()=std::move(result.residual);
    output.gate.dense()=std::move(result.gate);
    return output;
}
Tensor Residual::post(const Tensor& input, const Tensor& residual, const Tensor& gate) const {
    Tensor output{DenseTensor{}};
    if (!impl_) throw std::logic_error("gated-residual operator is not initialized");
    const MfqCudaGuard guard(residual.device());
    output.dense()=impl_->core.post(input.dense(),residual.dense(),gate.dense());
    return output;
}
bool Residual::can_chain(const Tensor& input) const {
    if (!impl_) return false;
    const MfqCudaGuard guard(input.device());
    return impl_->core.can_fuse(input.dense());
}
} // namespace tyloi::ops
#endif
