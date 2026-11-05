#pragma once

#include "engine/models/minicpmo45/layer.h"
#include "engine/models/common/model_config.h"
#include "engine/ops.h"
#include "engine/models/common/causal_metadata.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/backends/cuda/executor/task.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/models/common/model_graph.h"
#include "engine/models/common/capabilities.h"
#include "engine/models/source.h"

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

struct LayerPlacementConfig;
namespace tyloi::cuda::weight_loader { struct Loader; }

namespace tyloi::cuda::minicpmo45 {

// The standalone TTS checkpoint uses offset RMS weights, unlike composite Tts.
struct Model {
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    CudaModelPlan plan;
    tyloi::models::ModelConfig config;
    tyloi::models::CausalLmMetadata metadata;
    RopeCache rope, cpu_rope;
    std::unordered_map<int, RopeCache> device_ropes;
    std::shared_ptr<const ::tyloi::Tensor> embed, lm_head;
    DenseTensor output_norm;
    std::vector<Layer> layers;
};

Model read(std::shared_ptr<const tyloi::ModelSource>, std::string_view definition, int64_t context);
void load(Model&, const weight_loader::Loader&, const LayerPlacementConfig&, int cpu_layers,
          const tyloi::ops::YarnScaling&);

} // namespace tyloi::cuda::minicpmo45
