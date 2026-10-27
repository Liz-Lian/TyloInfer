#pragma once

#include "native/cuda/tensor/backend.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace mfq::cuda {

struct Input {
    mfq_tensor_backend::Tensor ids;
    std::optional<mfq_tensor_backend::Tensor> embeddings;
    std::optional<mfq_tensor_backend::Tensor> positions;
    std::optional<mfq_tensor_backend::Tensor> lengths;
    std::optional<mfq_tensor_backend::Tensor> attention_mask;
    std::optional<mfq_tensor_backend::Tensor> cache_positions;
    std::vector<mfq_tensor_backend::Tensor>* trace = nullptr;
    mfq_tensor_backend::Tensor* raw_hidden = nullptr;
    bool advance = false;
    std::int64_t confirmed = 0;
    std::int64_t kv_length = 0;
    std::int64_t attention_parts = 0;
};

} // namespace mfq::cuda
