#pragma once

#include "engine/backends/cuda/native/profile.h"
#include "backend.h"
#include "engine/weights/vq.h"

#include <cstdint>
#include <utility>
#include <vector>

mfq_tensor_backend::Tensor nvq_matmul_multi2(
    CudaProfiler& profiler,
    const NvqWeight& first,
    const NvqWeight& second,
    mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor nvq_matmul_swiglu(
    CudaProfiler& profiler,
    const NvqWeight& gate,
    const NvqWeight& up,
    mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor nvq_matmul(
    CudaProfiler& profiler,
    const NvqWeight& weight,
    mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor nvq_matmul_input_mul(
    CudaProfiler& profiler,
    const NvqWeight& weight,
    mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor gate,
    int mode);

bool nvq_pair_compatible(const NvqWeight& first, const NvqWeight& second);
mfq_tensor_backend::Tensor nvq_dequant(const NvqWeight& weight);
mfq_tensor_backend::Tensor nvq_embedding(
    const NvqWeight& weight, mfq_tensor_backend::Tensor token_ids);
