#pragma once

#include "engine/tensor.h"
#include "engine/ops.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tyloi::models {

struct Input {
    using Trace = std::vector<Tensor>;

    Tensor ids, embeddings, positions, lengths, cache_positions;
    Trace* trace = nullptr;
    Tensor* raw_hidden = nullptr;
    bool advance = false;
    std::int64_t confirmed = 0;
    std::int64_t kv_length = 0;
    std::int64_t attention_parts = 0;
};

struct ForwardOptions {
    std::int64_t kv_capacity = 0;
    ops::Trace* trace = nullptr;
    int trace_layer = -1;
    int primary = 0;
};

struct Prepared {
    std::vector<std::int64_t> token_ids;
    Tensor embeddings;
    Tensor positions;
    int decode_position_delta = 0;
    std::string cache_key;
    Tensor image_mask;

    bool transformed() const noexcept {
        return (embeddings && embeddings.defined()) ||
            (positions && positions.defined()) ||
            decode_position_delta != 0;
    }
};

} // namespace tyloi::models
