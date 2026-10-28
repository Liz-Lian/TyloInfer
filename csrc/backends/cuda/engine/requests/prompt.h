#pragma once

#include "backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mfq::cuda {

struct PreparedPrompt {
    std::vector<std::int64_t> token_ids;
    mfq_tensor_backend::Tensor embeddings;
    mfq_tensor_backend::Tensor positions;
    int decode_position_delta = 0;
    std::string cache_key;
    mfq_tensor_backend::Tensor image_mask;

    bool transformed() const noexcept {
        return embeddings.defined() || positions.defined() ||
            decode_position_delta != 0;
    }
};

} // namespace mfq::cuda
