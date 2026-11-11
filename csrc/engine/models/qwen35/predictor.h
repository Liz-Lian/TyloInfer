#pragma once

#include "engine/models/qwen35/config.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/models/common/attention.h"

#include <vector>

namespace tyloi::cuda::qwen35 {

struct Predictor {
    struct Layer {
        FullAttention attention;
#ifdef TYLOINFER_NATIVE_CUDA
        ::tyloi::Tensor ffn_norm;
#else
        ::tyloi::DenseTensor ffn_norm;
#endif
        Ffn ffn;
    };

    tyloi::models::qwen35::Config config;
    ::tyloi::Tensor fusion;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor hidden_norm, embedding_norm, output_norm;
#else
    ::tyloi::DenseTensor hidden_norm, embedding_norm, output_norm;
#endif
    std::vector<Layer> layers;
};

} // namespace tyloi::cuda::qwen35
