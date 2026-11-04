#pragma once

#include "engine/tensor.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace tyloi::models {

struct Input {
    using Tensor = tyloi::Tensor;
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

} // namespace tyloi::models
