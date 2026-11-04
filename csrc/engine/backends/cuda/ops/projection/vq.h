#pragma once

#include "engine/backends/cuda/native/profile.h"
#include "backend.h"
#include "engine/weights/vq.h"

#include <cstdint>
#include <utility>
#include <vector>

::tyloi::Tensor nvq_matmul_multi2(
    CudaProfiler& profiler,
    const NvqWeight& first,
    const NvqWeight& second,
    ::tyloi::Tensor input);
::tyloi::Tensor nvq_matmul_swiglu(
    CudaProfiler& profiler,
    const NvqWeight& gate,
    const NvqWeight& up,
    ::tyloi::Tensor input);
::tyloi::Tensor nvq_matmul(
    CudaProfiler& profiler,
    const NvqWeight& weight,
    ::tyloi::Tensor input);
::tyloi::Tensor nvq_matmul_input_mul(
    CudaProfiler& profiler,
    const NvqWeight& weight,
    ::tyloi::Tensor input,
    ::tyloi::Tensor gate,
    int mode);

bool nvq_pair_compatible(const NvqWeight& first, const NvqWeight& second);
::tyloi::Tensor nvq_dequant(const NvqWeight& weight);
::tyloi::Tensor nvq_embedding(
    const NvqWeight& weight, ::tyloi::Tensor token_ids);
