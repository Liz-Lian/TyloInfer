#pragma once

#include "engine/weights/bias.h"
#include "engine/backends/cuda/ops/projection/projection.h"

#include <optional>
#include <utility>

namespace tyloi::cuda {

inline ::tyloi::Tensor biased(
        const Projector& projector, const QuantLinear& weight,
        const ::tyloi::Tensor& bias,
        ::tyloi::Tensor input) {
    if (weight.is_dense() && !weight.tensor_parallel()) {
        check_projection(weight, input);
        const auto output_dtype = input.scalar_type();
        const auto dense_dtype = weight.dense.scalar_type();
        std::optional<::tyloi::Tensor> dense_bias = std::nullopt;
        if (bias.defined()) {
            dense_bias = bias.to(dense_dtype);
        }
        return mfq_linear(input.to(dense_dtype), weight.dense, dense_bias)
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

inline ::tyloi::Tensor biased(
        const Projector& projector, const BiasWeight& value,
        ::tyloi::Tensor input) {
    return biased(projector, value.weight, value.bias, std::move(input));
}

inline ::tyloi::Tensor affine(
        const Projector& projector, const BiasWeight& value,
        ::tyloi::Tensor input) {
    const auto& weight = value.weight;
    const auto& bias = value.bias;

    auto output = projector.run(weight, std::move(input));
    return bias.defined()
        ? output + bias.to(output.scalar_type())
        : output;
}

} // namespace tyloi::cuda
