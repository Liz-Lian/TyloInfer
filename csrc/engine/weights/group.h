#pragma once

#include "engine/tensor.h"

#include <vector>

namespace tyloi::weights {

std::vector<::tyloi::Tensor> pack_projections(
    std::vector<::tyloi::Tensor> weights,
    bool group_nint,
    bool cpu_layer);

#if TYLOINFER_ENABLE_MFQ_COMPAT
std::vector<::tyloi::Tensor> pack_nint(
    std::vector<::tyloi::Tensor> weights);
#endif

} // namespace tyloi::weights
