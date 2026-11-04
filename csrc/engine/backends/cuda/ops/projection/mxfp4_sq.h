#pragma once

#include "engine/weights/sq.h"
#include "engine/backends/cuda/kernels/mxfp4_sq.h"

::tyloi::Tensor sq_matmul_cpu(
    const Mxfp4SqWeight& weight,
    ::tyloi::Tensor input);

::tyloi::Tensor dequant_mxfp4_sq_cpu(
    const Mxfp4SqWeight& weight);

namespace tyloi::cuda {
::tyloi::Tensor sq_matmul(const Mxfp4SqWeight& weight, ::tyloi::Tensor x);
}
