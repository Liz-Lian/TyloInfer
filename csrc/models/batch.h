#pragma once

#include "models/qwen35/batch_ops.h"
#include "models/qwen4_exp/batch.h"

#include <type_traits>

namespace mfq::cuda {

template <class Model>
using CudaBatchOperations = std::conditional_t<
    std::is_same_v<Model, Qwen4CausalLm>, qwen4_exp::BatchOps, qwen35::BatchOps>;

} // namespace mfq::cuda
