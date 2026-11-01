#pragma once

#include "engine/models/qwen35/config.h"
#include "engine/backends/cuda/executor/task.h"
#include "engine/models/common/ffn.h"
#include "engine/backends/cuda/ops/attention/full.h"

#include <vector>

namespace mfq::cuda::qwen35 {

struct Predictor {
    struct Layer {
        FullAttention attention;
        mfq_tensor_backend::Tensor ffn_norm;
        Ffn ffn;
    };

    mfq::models::qwen35::Config config;
    QuantLinear fusion;
    mfq_tensor_backend::Tensor hidden_norm, embedding_norm, output_norm;
    std::vector<Layer> layers;

    std::vector<Task> tasks() const;
};

} // namespace mfq::cuda::qwen35
