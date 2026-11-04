#pragma once
#include "backend.h"
#include <cstdint>

::tyloi::DenseTensor mxfp8_groupwise_gemm_f16_cuda(
    ::tyloi::DenseTensor values,::tyloi::DenseTensor scales,
    ::tyloi::DenseTensor input,std::int64_t groups);
::tyloi::DenseTensor mxfp8_groupwise_gemm_f32_cuda(
    ::tyloi::DenseTensor values,::tyloi::DenseTensor scales,
    ::tyloi::DenseTensor input,std::int64_t groups);
