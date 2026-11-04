#pragma once

#include "engine/backends/cuda/native/profile.h"
#include "backend.h"
#include "engine/weights/mx.h"

#include <cstdint>
#include <vector>

::tyloi::Tensor mxfp8_groupwise_matmul(
    CudaProfiler& profiler,
    const Mxfp8Weight& weight,
    ::tyloi::Tensor input,
    std::int64_t groups);
::tyloi::Tensor mxfp8_groupwise_matmul_f32(
    CudaProfiler& profiler,
    const Mxfp8Weight& weight,
    ::tyloi::Tensor input,
    std::int64_t groups);
::tyloi::Tensor mxfp8_matmul(
    CudaProfiler& profiler,
    const Mxfp8Weight& weight,
    ::tyloi::Tensor input);
::tyloi::Tensor mxfp8_matmul_f32(
    CudaProfiler& profiler,
    const Mxfp8Weight& weight,
    ::tyloi::Tensor input);
::tyloi::Tensor mxfp4_matmul(
    const Mxfp4Weight& weight, ::tyloi::Tensor input);

::tyloi::Tensor mxfp8_cpu_reference(const Mxfp8Weight& weight);
