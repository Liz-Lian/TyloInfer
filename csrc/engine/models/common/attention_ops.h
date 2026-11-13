#pragma once

#include "backend.h"
#include "engine/backends/cuda/ops/projection/projection.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace tyloi::cuda::attention_ops {
using DenseTensor = ::tyloi::DenseTensor;
using Linear = ::tyloi::Tensor;

inline DenseTensor rms_norm(const DenseTensor& value,
                            const ::tyloi::Tensor& weight_tensor, double eps) {
    auto f = value.to(::tyloi::kFloat32);
    const auto& weight = weight_tensor.dense();
    return (f * ::tyloi::cuda::rsqrt((f * f).mean(-1, true) + eps) * weight.to(::tyloi::kFloat32))
        .to(value.scalar_type());
}

} // namespace tyloi::cuda::attention_ops
