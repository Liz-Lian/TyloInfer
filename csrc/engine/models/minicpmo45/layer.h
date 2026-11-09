#pragma once

#include "engine/models/common/attention.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"

namespace tyloi::cuda::minicpmo45 {
using DenseTensor = ::tyloi::DenseTensor;

struct Layer {
    FullAttention attention;
#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<const ::tyloi::Tensor> ffn_norm;
#else
    DenseTensor ffn_norm;
#endif
    Ffn ffn;
    int device = 0;
    bool cpu = false;
};
} // namespace tyloi::cuda::minicpmo45
