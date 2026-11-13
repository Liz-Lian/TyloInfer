#pragma once

#include "engine/models/qwen35/config.h"
#include "engine/models/metadata.h"
#include "engine/models/causal_model.h"
#include "engine/ops/ffn.h"
#include "engine/ops/linear.h"
#include "engine/models/source/model_graph.h"
#include "engine/models/metadata.h"
#include "engine/models/common/attention.h"
#include "engine/models/source.h"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tyloi::cuda::qwen35 {

struct Components;

using DenseTensor = ::tyloi::DenseTensor;
using Config = tyloi::models::qwen35::Config;

struct Full {
    FullAttention attention;
    ::tyloi::Tensor ffn_norm;
    tyloi::ops::FfnWeights ffn;
};

struct Linear {
    std::unique_ptr<tyloi::ops::Linear> attention;
    tyloi::ops::LinearConfig geometry;
    ::tyloi::Tensor ffn_norm;
    tyloi::ops::FfnWeights ffn;
};

struct Layer {
    std::variant<Full, Linear> weights;
    int device = 0;
    bool cpu = false;
};

struct Model
    : tyloi::models::CausalModel
{
    /** Create an empty movable model before its weights are loaded. */
    Model();
    ~Model() override;
    Model(Model&&) noexcept;
    Model& operator=(Model&&) noexcept;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    ::tyloi::models::Plan plan;
    Config config;
    tyloi::models::CausalLmMetadata metadata;
    tyloi::ops::RopeTable rope, cpu_rope;
    std::unordered_map<int, tyloi::ops::RopeTable> device_ropes;
    ::tyloi::Tensor embed;
    ::tyloi::Tensor lm_head;
    ::tyloi::Tensor output_norm;
    std::vector<Layer> layers;
    int primary_device = 0;
    int cpu_layers = 0;
    bool transposed = false;
    std::unique_ptr<Components> components;
    /** Describe the assembled text model and its execution capabilities. */
    tyloi::models::Info info() const override;
    /** Report capabilities of the assembled text, vision and predictor weights. */
    MfqModelCapabilities capabilities() const override;
    /** Report component and KV capability metrics. */
    tyloi::engine::Metrics metrics() const override;
    /** Borrow this model's predictor, if assembled. */
    tyloi::models::Mtp* mtp() const noexcept override;
    /** Create stable-slot batch computation and retire idle startup KV when configured. */
    std::unique_ptr<tyloi::models::Batch> batch(
        const tyloi::models::BatchConfig&, tyloi::models::State* serial = nullptr) override;
    /** Create request-owned attention, media and predictor state. */
    std::unique_ptr<tyloi::models::State> make_state() const override;
    /** Prepare supported media using the requesting state's cache. */
    tyloi::StepSequence<tyloi::models::Prepared> prepare(
        const std::vector<std::int64_t>& tokens, const MfqMultimodalInput& media,
        tyloi::models::State& state) const override;
    /** Embed IDs on the requested primary device. */
    ::tyloi::Tensor embedding(const tyloi::Tensor& ids, int primary = 0) const override;
    /** Normalize final hidden states for the supplied batch and token shape. */
    ::tyloi::Tensor normalize(const tyloi::Tensor&, std::int64_t batch, std::int64_t tokens) const override;
    /** Apply the model's configured logit soft cap. */
    ::tyloi::Tensor softcap(const tyloi::Tensor&) const override;
    /** Project hidden states to vocabulary logits. */
    ::tyloi::Tensor logits(const tyloi::Tensor&, bool capped = true) const override;
    /** Execute a request step using state created by this model. */
    ::tyloi::Tensor forward(tyloi::models::Input, tyloi::models::State&,
                            const tyloi::models::ForwardOptions& = {}) const override;
};

} // namespace tyloi::cuda::qwen35
