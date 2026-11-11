#pragma once

#include "engine/models/common/attention.h"
#include "engine/models/common/mlp.h"

namespace tyloi::cuda::minicpmo45 {
using DenseTensor = ::tyloi::DenseTensor;

struct Layer {
    FullAttention attention;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor ffn_norm;
#else
    DenseTensor ffn_norm;
#endif
    tyloi::models::Mlp ffn;
    int device = 0;
    bool cpu = false;
};
} // namespace tyloi::cuda::minicpmo45
