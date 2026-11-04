#pragma once

#include "engine/weights/sq.h"
#include "engine/backends/cuda/kernels/fp8_sq.h"

::tyloi::Tensor sq_matmul_cpu(
    const Fp8SqWeight& weight,
    ::tyloi::Tensor input);

namespace tyloi::cuda {
::tyloi::Tensor sq_matmul(const Fp8SqWeight& weight, ::tyloi::Tensor x);
}

::tyloi::Tensor dequant_fp8_sq(
    const Fp8SqWeight& weight, bool fp32);
