#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

::tyloi::DenseTensor silu_mul_cuda(::tyloi::DenseTensor gate, ::tyloi::DenseTensor up);
::tyloi::DenseTensor gelu_mul_cuda(::tyloi::DenseTensor gate, ::tyloi::DenseTensor up);
::tyloi::DenseTensor clipped_silu_mul_cuda(::tyloi::DenseTensor gate,
    ::tyloi::DenseTensor up,double limit);
// relu(x)^2 for f32/f16/bf16 input; the output keeps the input dtype.
::tyloi::DenseTensor relu_squared_cuda(::tyloi::DenseTensor input);
