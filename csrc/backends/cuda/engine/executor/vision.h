#pragma once

#include "execution.h"
#include "backends/cuda/engine/weights/moe/expert_cache.h"

#include <cstddef>

namespace mfq::cuda::vision {

using Tensor = mfq_tensor_backend::Tensor;

template <class Forward>
Tensor vision_workspace(
    CudaExecutionContext& execution,
    std::size_t bytes,
    Forward forward) {
    MoePrefillWorkspaceLoan loan(execution.stores.weights.moe);
    const bool borrowed = loan.begin(bytes) != 0;
    auto result = forward();
    if (borrowed) {
        loan.suspend();
        result = result.clone();
        loan.finish();
    }
    return result;
}

} // namespace mfq::cuda::vision
