#pragma once

#include "engine/weights/sq.h"
#include "engine/backends/cuda/kernels/mxfp4_sq.h"

::tyloi::DenseTensor sq_matmul_cpu(
    const tyloi::Mxfp4SqTensor& weight,
    ::tyloi::DenseTensor input);

::tyloi::DenseTensor dequant_mxfp4_sq_cpu(
    const tyloi::Mxfp4SqTensor& weight);

namespace tyloi::cuda {
::tyloi::DenseTensor sq_matmul(const tyloi::Mxfp4SqTensor& weight, ::tyloi::DenseTensor x);
}
