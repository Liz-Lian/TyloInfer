#pragma once

#include "engine/tensor.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace tyloi::models {

struct Input {
    using DenseTensor = tyloi::DenseTensor;
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

} // namespace tyloi::models
