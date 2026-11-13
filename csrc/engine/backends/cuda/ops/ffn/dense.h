#pragma once

#include "mlp.h"

#include <optional>

namespace tyloi::cuda {

struct DenseOptions {
    Mlp::Options mlp;
    bool important_parallel = true;
    bool important_fp32 = false;
    bool fp32_residual = false;
    bool bf16_norm = true;
    bool bf16_swiglu = true;
    bool bf16_residual = true;
};

} // namespace tyloi::cuda

namespace tyloi::cuda::ops {

tyloi::DenseTensor mlp(
    tyloi::DenseTensor input, std::span<const ::tyloi::Tensor> gate_up,
    const ::tyloi::Tensor& down, tyloi::ops::MlpActivation activation,
    double swiglu_limit);

tyloi::DenseTensor bf16_mlp(
    tyloi::DenseTensor input, std::span<const ::tyloi::Tensor> gate_up,
    const ::tyloi::Tensor& down, tyloi::ops::MlpActivation activation,
    double swiglu_limit);

tyloi::DenseTensor additive_mlp(
    tyloi::DenseTensor input, std::span<const ::tyloi::Tensor> gate_up,
    const ::tyloi::Tensor& down, tyloi::ops::MlpActivation activation,
    double swiglu_limit, std::span<const ::tyloi::Tensor> extra_gate_up,
    const ::tyloi::Tensor& extra_down, tyloi::ops::MlpActivation extra_activation,
    double extra_limit);

std::optional<tyloi::DenseTensor> mlp_residual(
    tyloi::DenseTensor input, tyloi::DenseTensor residual,
    std::span<const ::tyloi::Tensor> gate_up, const ::tyloi::Tensor& down,
    tyloi::ops::MlpActivation activation, double swiglu_limit,
    bool recurrent);

} // namespace tyloi::cuda::ops
