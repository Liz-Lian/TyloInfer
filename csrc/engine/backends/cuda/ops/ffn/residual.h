#pragma once

#include "engine/tensor.h"

#include <array>

#ifdef TYLOINFER_NATIVE_CUDA
namespace tyloi::cuda::ops {

std::array<::tyloi::DenseTensor, 2> residual_norm(
    ::tyloi::DenseTensor residual, ::tyloi::DenseTensor value,
    const ::tyloi::Tensor& weight, double eps, double offset, bool bf16);

std::array<::tyloi::DenseTensor, 2> recurrent_norm(
    ::tyloi::DenseTensor residual, ::tyloi::DenseTensor value,
    const ::tyloi::Tensor& weight, double eps, double offset);

::tyloi::DenseTensor residual_add(
    ::tyloi::DenseTensor residual, ::tyloi::DenseTensor value);

::tyloi::DenseTensor recurrent_add(
    ::tyloi::DenseTensor residual, ::tyloi::DenseTensor value);

} // namespace tyloi::cuda::ops
#endif
