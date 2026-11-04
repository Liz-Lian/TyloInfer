#pragma once

#include "engine/models/input.h"

#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/backends/cuda/native/tensor/backend.h"
#endif

namespace tyloi::cuda {

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
using Input = tyloi::models::Input;
#else
struct Input {
    using DenseTensor = ::tyloi::DenseTensor;
    using Trace = std::vector<DenseTensor>;

    DenseTensor ids;
    std::optional<DenseTensor> embeddings;
    std::optional<DenseTensor> positions;
    std::optional<DenseTensor> lengths;
    std::optional<DenseTensor> attention_mask;
    std::optional<DenseTensor> cache_positions;
    Trace* trace = nullptr;
    DenseTensor* raw_hidden = nullptr;
    bool advance = false;
    std::int64_t confirmed = 0;
    std::int64_t kv_length = 0;
    std::int64_t attention_parts = 0;
};
#endif

} // namespace tyloi::cuda
