#pragma once

#include "engine/models/input.h"

#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/backends/cuda/native/tensor/backend.h"
#endif

namespace mfq::cuda {

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
using Input = mfq::models::Input;
#else
struct Input {
    using Tensor = mfq_tensor_backend::Tensor;
    using Trace = std::vector<Tensor>;

    Tensor ids;
    std::optional<Tensor> embeddings;
    std::optional<Tensor> positions;
    std::optional<Tensor> lengths;
    std::optional<Tensor> attention_mask;
    std::optional<Tensor> cache_positions;
    Trace* trace = nullptr;
    Tensor* raw_hidden = nullptr;
    bool advance = false;
    std::int64_t confirmed = 0;
    std::int64_t kv_length = 0;
    std::int64_t attention_parts = 0;
};
#endif

} // namespace mfq::cuda
