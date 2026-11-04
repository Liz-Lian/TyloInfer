#pragma once

#include "engine/backends/cuda/native/profile.h"
#include "backend.h"
#include "engine/weights/vq.h"

#include <cstdint>
#include <utility>
#include <vector>

::tyloi::DenseTensor nvq_matmul_multi2(
    CudaProfiler& profiler,
    const tyloi::NvqTensor& first,
    const tyloi::NvqTensor& second,
    ::tyloi::DenseTensor input);
::tyloi::DenseTensor nvq_matmul_swiglu(
    CudaProfiler& profiler,
    const tyloi::NvqTensor& gate,
    const tyloi::NvqTensor& up,
    ::tyloi::DenseTensor input);
::tyloi::DenseTensor nvq_matmul(
    CudaProfiler& profiler,
    const tyloi::NvqTensor& weight,
    ::tyloi::DenseTensor input);
::tyloi::DenseTensor nvq_matmul_input_mul(
    CudaProfiler& profiler,
    const tyloi::NvqTensor& weight,
    ::tyloi::DenseTensor input,
    ::tyloi::DenseTensor gate,
    int mode);

bool nvq_pair_compatible(const tyloi::NvqTensor& first, const tyloi::NvqTensor& second);
::tyloi::DenseTensor nvq_dequant(const tyloi::NvqTensor& weight);
::tyloi::DenseTensor nvq_embedding(
    const tyloi::NvqTensor& weight, ::tyloi::DenseTensor token_ids);
