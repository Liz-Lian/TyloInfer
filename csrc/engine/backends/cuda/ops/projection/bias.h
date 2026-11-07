#pragma once

#include "engine/weights/bias.h"
#include "engine/backends/cuda/ops/projection/projection.h"

#include <optional>
#include <utility>

namespace tyloi::cuda {

inline ::tyloi::DenseTensor biased(
        const Projector& projector, const ::tyloi::Tensor& weight,
        const ::tyloi::DenseTensor* bias,
        ::tyloi::DenseTensor input) {
    const auto* dense = dynamic_cast<const ::tyloi::DenseTensor*>(&weight);
    if (dense != nullptr) {
        check_projection(weight, input);
        const auto output_dtype = input.scalar_type();
        const auto dense_dtype = dense->scalar_type();
        std::optional<::tyloi::DenseTensor> dense_bias = std::nullopt;
        if (bias && bias->defined()) {
            dense_bias = bias->to(dense_dtype);
        }
        return mfq_linear(input.to(dense_dtype), *dense, dense_bias)
            .to(output_dtype)
            .contiguous();
    }
    auto output = projector.run(weight, input);
    const auto output_dtype = input.scalar_type();
#ifdef TYLOINFER_NATIVE_CUDA
    std::optional<MfqCudaGuard> guard;
    if (output.is_cuda()) guard.emplace(output.device().index);
#endif
    if (bias && bias->defined()) {
        output = output + bias->to(output.scalar_type());
    }
    return output.to(output_dtype).contiguous();
}

inline ::tyloi::DenseTensor biased(
        const Projector& projector, const weights::BiasWeight& value,
        ::tyloi::DenseTensor input) {
    const auto* bias = value.bias ? &dynamic_cast<const ::tyloi::DenseTensor&>(*value.bias) : nullptr;
    return biased(projector, *value.weight, bias, std::move(input));
}

inline ::tyloi::DenseTensor affine(
        const Projector& projector, const weights::BiasWeight& value,
        ::tyloi::DenseTensor input) {
    const auto* bias = value.bias ? &dynamic_cast<const ::tyloi::DenseTensor&>(*value.bias) : nullptr;

    auto output = projector.run(*value.weight, std::move(input));
    return bias && bias->defined()
        ? output + bias->to(output.scalar_type())
        : output;
}

} // namespace tyloi::cuda
