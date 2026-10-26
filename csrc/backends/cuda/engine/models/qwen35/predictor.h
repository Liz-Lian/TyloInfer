#pragma once

#include "engine/models/qwen35/config.h"
#include "engine/task.h"
#include "engine/models/ffn.h"
#include "ops/attention/full.h"

#include <optional>
#include <vector>

struct TransformerWeightLoader;

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

    static std::optional<Predictor> load(TransformerWeightLoader& loader,
        const mfq::models::qwen35::Config& config, KvQuantization quantization);
    std::vector<Task> tasks() const;
};

} // namespace mfq::cuda::qwen35
