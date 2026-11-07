#pragma once

#include "engine/tensor.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace tyloi::models {

struct Input {
    using Trace = std::vector<std::shared_ptr<Tensor>>;

    std::shared_ptr<const Tensor> ids, embeddings, positions, lengths, cache_positions;
    Trace* trace = nullptr;
    std::shared_ptr<Tensor>* raw_hidden = nullptr;
    bool advance = false;
    std::int64_t confirmed = 0;
    std::int64_t kv_length = 0;
    std::int64_t attention_parts = 0;
};

} // namespace tyloi::models
