#pragma once

#include "engine/backends/cuda/native/profile.h"
#include "backend.h"
#include "engine/weights/mx.h"

#include <cstdint>
#include <vector>

::tyloi::DenseTensor mxfp8_groupwise_matmul(
    CudaProfiler& profiler,
    const tyloi::Mxfp8Tensor& weight,
    ::tyloi::DenseTensor input,
    std::int64_t groups);
::tyloi::DenseTensor mxfp8_groupwise_matmul_f32(
    CudaProfiler& profiler,
    const tyloi::Mxfp8Tensor& weight,
    ::tyloi::DenseTensor input,
    std::int64_t groups);
::tyloi::DenseTensor mxfp8_matmul(
    CudaProfiler& profiler,
    const tyloi::Mxfp8Tensor& weight,
    ::tyloi::DenseTensor input);
::tyloi::DenseTensor mxfp8_matmul_f32(
    CudaProfiler& profiler,
    const tyloi::Mxfp8Tensor& weight,
    ::tyloi::DenseTensor input);
::tyloi::DenseTensor mxfp4_matmul(
    const tyloi::Mxfp4Tensor& weight, ::tyloi::DenseTensor input);

::tyloi::DenseTensor mxfp8_cpu_reference(const tyloi::Mxfp8Tensor& weight);
