#pragma once

#include "engine/ops.h"

namespace tyloi::cuda::ops {

tyloi::ops::SelectedExperts topk(
    tyloi::DenseTensor logits, std::int64_t count,
    const tyloi::ops::MoeRouting& routing,
    const tyloi::DenseTensor* bias);
tyloi::DenseTensor expert_scale(
    tyloi::DenseTensor weights, tyloi::DenseTensor ids,
    const tyloi::DenseTensor& scales);
tyloi::DenseTensor weighted_reduce(
    tyloi::DenseTensor pairs, tyloi::DenseTensor weights);
tyloi::ops::RoutedGeglu routed_geglu(
    tyloi::DenseTensor input, tyloi::DenseTensor ids,
    const ::tyloi::Tensor& gate_up, const ::tyloi::Tensor& down,
    bool trace);

} // namespace tyloi::cuda::ops
