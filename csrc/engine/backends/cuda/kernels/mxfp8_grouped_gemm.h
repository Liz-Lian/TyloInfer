#pragma once
#include "backend.h"
#include <cstdint>

::tyloi::Tensor mxfp8_groupwise_gemm_f16_cuda(
    ::tyloi::Tensor values,::tyloi::Tensor scales,
    ::tyloi::Tensor input,std::int64_t groups);
::tyloi::Tensor mxfp8_groupwise_gemm_f32_cuda(
    ::tyloi::Tensor values,::tyloi::Tensor scales,
    ::tyloi::Tensor input,std::int64_t groups);
