#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

::tyloi::Tensor silu_mul_cuda(::tyloi::Tensor gate, ::tyloi::Tensor up);
::tyloi::Tensor gelu_mul_cuda(::tyloi::Tensor gate, ::tyloi::Tensor up);
::tyloi::Tensor clipped_silu_mul_cuda(::tyloi::Tensor gate,
    ::tyloi::Tensor up,double limit);
// relu(x)^2 for f32/f16/bf16 input; the output keeps the input dtype.
::tyloi::Tensor relu_squared_cuda(::tyloi::Tensor input);
