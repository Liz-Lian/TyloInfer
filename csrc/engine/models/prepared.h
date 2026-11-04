#pragma once

#include "engine/tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tyloi::models {

struct Prepared {
    std::vector<std::int64_t> token_ids;
    DenseTensor embeddings;
    DenseTensor positions;
    int decode_position_delta = 0;
    std::string cache_key;
    DenseTensor image_mask;

    bool transformed() const noexcept {
        return embeddings.defined() || positions.defined() ||
            decode_position_delta != 0;
    }
};

} // namespace tyloi::models
