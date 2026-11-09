#pragma once

#include "engine/tensor.h"

#include <memory>
#include <vector>

namespace tyloi::weights {

std::vector<std::shared_ptr<const ::tyloi::Tensor>> pack_projections(
    std::vector<std::shared_ptr<const ::tyloi::Tensor>> weights,
    bool group_nint,
    bool cpu_layer);

#if TYLOINFER_ENABLE_MFQ_COMPAT
std::vector<std::shared_ptr<const ::tyloi::Tensor>> pack_nint(
    std::vector<std::shared_ptr<const ::tyloi::Tensor>> weights);
#endif

} // namespace tyloi::weights
