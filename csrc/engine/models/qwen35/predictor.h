#pragma once

#include "engine/models/qwen35/config.h"
#include "engine/ops/ffn.h"
#include "engine/models/common/attention.h"

#include <vector>

namespace tyloi::cuda::qwen35 {

struct Predictor {
    struct Layer {
        FullAttention attention;
        ::tyloi::Tensor ffn_norm;
        tyloi::ops::FfnWeights ffn;
    };

    tyloi::models::qwen35::Config config;
    ::tyloi::Tensor fusion;
    ::tyloi::Tensor hidden_norm, embedding_norm, output_norm;
    std::vector<Layer> layers;
};

} // namespace tyloi::cuda::qwen35
