#pragma once

#include "engine/models/qwen35/config.h"
#include "engine/backends/cuda/executor/task.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/models/common/attention.h"

#include <vector>

namespace tyloi::cuda::qwen35 {

struct Predictor {
    struct Layer {
        FullAttention attention;
        ::tyloi::DenseTensor ffn_norm;
        Ffn ffn;
    };

    tyloi::models::qwen35::Config config;
    QuantLinear fusion;
    ::tyloi::DenseTensor hidden_norm, embedding_norm, output_norm;
    std::vector<Layer> layers;

    std::vector<Task> tasks() const;
};

} // namespace tyloi::cuda::qwen35
