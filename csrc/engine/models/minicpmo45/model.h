#pragma once

#include "engine/models/minicpmo45/layer.h"
#include "engine/models/source/model_config.h"
#include "engine/ops.h"
#include "engine/models/metadata.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/models/causal_model.h"
#include "engine/models/load.h"
#endif
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/models/source/model_graph.h"
#include "engine/models/metadata.h"
#include "engine/models/source.h"

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

struct LayerPlacementConfig;
namespace tyloi::weights { struct Loader; }

namespace tyloi::cuda::minicpmo45 {

// The standalone TTS checkpoint uses offset RMS weights, unlike composite Tts.
struct Model
#ifdef TYLOINFER_NATIVE_CUDA
    : tyloi::models::CausalModel
#endif
{
#ifdef TYLOINFER_NATIVE_CUDA
    /** Create an empty movable model before loading weights. */
    Model();
    ~Model() override;
    Model(Model&&) noexcept;
    Model& operator=(Model&&) noexcept;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
#endif
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    ::tyloi::models::Plan plan;
    tyloi::models::ModelConfig config;
    tyloi::models::CausalLmMetadata metadata;
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::ops::RopeTable rope, cpu_rope;
    std::unordered_map<int, tyloi::ops::RopeTable> device_ropes;
#else
    RopeCache rope, cpu_rope;
    std::unordered_map<int, RopeCache> device_ropes;
#endif
    ::tyloi::Tensor embed, lm_head;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor output_norm;
#else
    DenseTensor output_norm;
#endif
    std::vector<Layer> layers;
#ifdef TYLOINFER_NATIVE_CUDA
    int primary_device = 0, cpu_layers = 0;
    /** Describe model dimensions and execution capabilities. */
    tyloi::models::Info info() const override;
    /** Report assembled model capabilities. */
    MfqModelCapabilities capabilities() const override;
    /** Report model-owned resource metrics. */
    tyloi::engine::Metrics metrics() const override;
    /** Create request-owned state for this model. */
    std::unique_ptr<tyloi::models::State> make_state() const override;
    /** Prepare supported multimodal inputs for this request. */
    tyloi::StepSequence<tyloi::models::Prepared> prepare(const std::vector<std::int64_t>&,
        const MfqMultimodalInput&, tyloi::models::State&) const override;
    /** Embed token IDs on the requested primary device. */
    Tensor embedding(const Tensor&, int primary = 0) const override;
    /** Apply the output normalization. */
    Tensor normalize(const Tensor&, std::int64_t batch, std::int64_t tokens) const override;
    /** Apply this model's logit soft cap. */
    Tensor softcap(const Tensor&) const override;
    /** Project normalized hidden states to logits. */
    Tensor logits(const Tensor&, bool capped = true) const override;
    /** Run a request step with explicit input and numerical policy. */
    Tensor forward(tyloi::models::Input, tyloi::models::State&,
                   const tyloi::models::ForwardOptions& = {}) const override;
#endif
};

#ifdef TYLOINFER_NATIVE_CUDA
/** Load standalone TTS weights using a resolved source and placement. */
std::unique_ptr<Model> load(std::shared_ptr<const tyloi::ModelSource>, std::string_view definition,
    const tyloi::weights::Loader&, LayerPlacementConfig&, const tyloi::models::LoadOptions&);
#endif
Model read(std::shared_ptr<const tyloi::ModelSource>, std::string_view definition, int64_t context);
void load(Model&, const tyloi::weights::Loader&, const LayerPlacementConfig&, int cpu_layers,
          const tyloi::ops::YarnScaling&);

} // namespace tyloi::cuda::minicpmo45
