#pragma once

#include "engine/models/minicpmo45/layer.h"
#include "engine/models/minicpmo45/config.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/models/metadata.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/models/causal_model.h"
#include "engine/models/load.h"
#include "engine/models/media.h"
#endif
#include "engine/models/metadata.h"
#include "engine/models/source.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace tyloi::cuda::minicpmo45 { class Components; }
namespace tyloi::cuda::minicpmo45::text {
using Config = tyloi::models::minicpmo45::Config;

struct Model
#ifdef TYLOINFER_NATIVE_CUDA
    : tyloi::models::CausalModel, tyloi::models::Media
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
    Config config;
    tyloi::models::CausalLmMetadata metadata;
    ::tyloi::Tensor embed, lm_head;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor output_norm;
#else
    DenseTensor output_norm;
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::ops::RopeTable rope, cpu_rope;
    std::unordered_map<int, tyloi::ops::RopeTable> device_ropes;
#else
    RopeCache rope, cpu_rope;
    std::unordered_map<int, RopeCache> device_ropes;
#endif
    std::vector<Layer> layers;
#ifdef TYLOINFER_NATIVE_CUDA
    int primary_device = 0, cpu_layers = 0;
    std::unique_ptr<::tyloi::cuda::minicpmo45::Components> components;
    /** Describe model dimensions and execution capabilities. */
    tyloi::models::Info info() const override;
    /** Report assembled model capabilities. */
    MfqModelCapabilities capabilities() const override;
    /** Report model-owned resource metrics. */
    tyloi::engine::Metrics metrics() const override;
    /** Create request-owned state for this model. */
    std::unique_ptr<tyloi::models::State> make_state() const override;
    /** Borrow available media and speech capabilities. */
    const tyloi::models::Media* media() const noexcept override;
    const tyloi::models::Speech* speech() const noexcept override;
    /** Encode media for the supplied explicit tensor inputs. */
    std::unique_ptr<tyloi::models::Media::Request> encode(tyloi::models::MediaInput) const override;
    /** Encode image or audio batches without prompt assembly. */
    std::unique_ptr<tyloi::models::Media::Request> images(Tensor, Tensor, Tensor) const override;
    std::unique_ptr<tyloi::models::Media::Request> audios(Tensor, Tensor) const override;
    /** Create a duplex session borrowing this model and request. */
    std::unique_ptr<tyloi::models::Duplex> duplex(tyloi::models::State&) const override;
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
    /** Project hidden states without the text adapter's BF16 output conversion. */
    Tensor project(const Tensor&) const override;
    /** Run a request step with explicit input and numerical policy. */
    Tensor forward(tyloi::models::Input, tyloi::models::State&,
                   const tyloi::models::ForwardOptions& = {}) const override;
#endif
};
} // namespace tyloi::cuda::minicpmo45::text
