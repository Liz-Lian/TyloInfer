#pragma once

#include "backend.h"
#include "engine/backends/cuda/ops/projection/projection.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace tyloi::cuda::attention_ops {
using DenseTensor = ::tyloi::DenseTensor;
using Linear = ::tyloi::Tensor;
using Transform = std::function<DenseTensor(const Projector&, const DenseTensor&)>;
using Embedding = std::function<DenseTensor(const DenseTensor&)>;

inline DenseTensor rms_norm(const DenseTensor& value,
#ifdef TYLOINFER_NATIVE_CUDA
                            const ::tyloi::Tensor& weight_tensor, double eps) {
#else
                            const DenseTensor& weight, double eps) {
#endif
    auto f = value.to(::tyloi::kFloat32);
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& weight = weight_tensor.dense();
#endif
    return (f * ::tyloi::cuda::rsqrt((f * f).mean(-1, true) + eps) * weight.to(::tyloi::kFloat32))
        .to(value.scalar_type());
}

} // namespace tyloi::cuda::attention_ops
