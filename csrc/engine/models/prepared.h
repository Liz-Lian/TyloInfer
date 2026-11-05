#pragma once

#include "engine/tensor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tyloi::models {

struct Prepared {
    std::vector<std::int64_t> token_ids;
    std::shared_ptr<const Tensor> embeddings;
    std::shared_ptr<const Tensor> positions;
    int decode_position_delta = 0;
    std::string cache_key;
    std::shared_ptr<const Tensor> image_mask;

    bool transformed() const noexcept {
        return (embeddings && embeddings->defined()) ||
            (positions && positions->defined()) ||
            decode_position_delta != 0;
    }
};

} // namespace tyloi::models
