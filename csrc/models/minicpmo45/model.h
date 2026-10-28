#pragma once

#include "engine/models/common/model_config.h"
#include "engine/models/common/causal_metadata.h"
#include "backends/cuda/engine/models/ffn.h"
#include "engine/task.h"
#include "formats/model_graph.h"
#include "cuda_model_plan.h"
#include "source/model_source.h"

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

struct LayerPlacementConfig;
namespace mfq::cuda::weight_loader { struct Loader; }

namespace mfq::cuda::minicpmo45 {

using Tensor = mfq_tensor_backend::Tensor;

struct Layer {
    FullAttention attention;
    Tensor ffn_norm;
    Ffn ffn;
    int device = 0;
    bool cpu = false;

    std::array<Task, 2> tasks() const;
};

// The standalone TTS checkpoint uses offset RMS weights, unlike composite Tts.
struct Model {
    std::shared_ptr<const mfq::ModelSource> source;
    mfq::ModelGraph graph;
    CudaModelPlan plan;
    mfq::models::ModelConfig config;
    mfq::models::CausalLmMetadata metadata;
    RopeCache rope, cpu_rope;
    std::unordered_map<int, RopeCache> device_ropes;
    QuantLinear embed, lm_head;
    Tensor output_norm;
    std::vector<Layer> layers;
};

Model read(std::shared_ptr<const mfq::ModelSource>, std::string_view definition, int64_t context);
void load(Model&, const weight_loader::Loader&, const LayerPlacementConfig&, int cpu_layers,
          const mfq::models::YarnScaling&);

} // namespace mfq::cuda::minicpmo45
