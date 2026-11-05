#pragma once

#include "engine/weights/bias.h"
#include "engine/backends/cuda/ops/projection/projection.h"

#include <optional>
#include <utility>

namespace tyloi::cuda {

inline ::tyloi::DenseTensor biased(
        const Projector& projector, const ::tyloi::Tensor& weight,
        const ::tyloi::DenseTensor& bias,
        ::tyloi::DenseTensor input) {
    const auto* dense = dynamic_cast<const ::tyloi::DenseTensor*>(&weight);
    if (dense != nullptr) {
        check_projection(weight, input);
        const auto output_dtype = input.scalar_type();
        const auto dense_dtype = dense->scalar_type();
        std::optional<::tyloi::DenseTensor> dense_bias = std::nullopt;
        if (bias.defined()) {
            dense_bias = bias.to(dense_dtype);
        }
        return mfq_linear(input.to(dense_dtype), *dense, dense_bias)
            .to(output_dtype)
            .contiguous();
    }
    auto output = projector.run(weight, input);
    const auto output_dtype = input.scalar_type();
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    std::optional<MfqCudaGuard> guard;
    if (output.is_cuda()) guard.emplace(output.device().index);
#endif
    if (bias.defined()) {
        output = output + bias.to(output.scalar_type());
    }
    return output.to(output_dtype).contiguous();
}

inline ::tyloi::DenseTensor biased(
        const Projector& projector, const BiasWeight& value,
        ::tyloi::DenseTensor input) {
    return biased(projector, *value.weight, value.bias, std::move(input));
}

inline ::tyloi::DenseTensor affine(
        const Projector& projector, const BiasWeight& value,
        ::tyloi::DenseTensor input) {
    const auto& bias = value.bias;

    auto output = projector.run(*value.weight, std::move(input));
    return bias.defined()
        ? output + bias.to(output.scalar_type())
        : output;
}

} // namespace tyloi::cuda
