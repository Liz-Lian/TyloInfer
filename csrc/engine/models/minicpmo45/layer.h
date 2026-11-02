#pragma once

#include "engine/backends/cuda/executor/task.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"

#include <array>

namespace mfq::cuda::minicpmo45 {
using Tensor = mfq_tensor_backend::Tensor;

struct Layer {
    FullAttention attention;
    Tensor ffn_norm;
    Ffn ffn;
    int device = 0;
    bool cpu = false;

    std::array<Task, 2> tasks() const;
};
} // namespace mfq::cuda::minicpmo45
