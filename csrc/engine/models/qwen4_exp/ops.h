#pragma once

#include "engine/ops.h"
#include "engine/ops/qsa.h"
#include "engine/models/causal_model.h"
#include "engine/models/batch.h"
#include "engine/models/metadata.h"
#include "engine/models/source.h"
#include "engine/metrics.h"
#include "engine/models/common/prompt.h"
#include "engine/models/qwen4_exp/layer.h"
#include "engine/models/session/size.h"
#include "engine/models/qwen4_exp/state.h"
#include "engine/models/qwen4_exp/config.h"
#include "engine/backends/cuda/ops/projection/quant_linear.h"

#include <memory>
#include <functional>
#include <cstdint>
#include <iosfwd>
#include <vector>

namespace tyloi::cuda::qwen4_exp { struct Layer; struct Components; }
namespace tyloi::models { struct Mtp; struct LoadOptions; }
namespace tyloi::weights { struct Loader; }
struct LayerPlacementConfig;

namespace tyloi::cuda {
struct Qwen4DecodeGraph;
class Qwen4BatchDecodeGraph;
// A request owns recurrent, positional and attention state; weights stay shared.
class Qwen4BatchRequestState
    : public tyloi::models::BatchState
{
  public:
    explicit Qwen4BatchRequestState(qwen4_exp::Model&);
    ~Qwen4BatchRequestState() override;
    Qwen4BatchRequestState(const Qwen4BatchRequestState&) = delete;
    Qwen4BatchRequestState& operator=(const Qwen4BatchRequestState&) = delete;
    /** Return the committed request position. */
    int64_t position() const override;
    /** Return request-owned resident cache bytes. */
    std::size_t resident_bytes() const override;
    /** Plan the backing needed before initialization. */
    tyloi::models::BatchGrowth initial_growth_bytes() const override;
    /** Initialize this request after admission. */
    void initialize() override;
    /** Plan persistent cache growth for the next token span. */
    tyloi::models::BatchGrowth prefill_growth_bytes(int64_t tokens) const override;
    /** Check the model that owns the request's batch layout. */
    bool belongs_to(const qwen4_exp::Model& model) const noexcept;
  private:
    struct Impl;
    std::unique_ptr<Impl> state_;
    friend class Qwen4BatchDecodeGraph;
    friend tyloi::StepSequence<std::vector<::tyloi::Tensor>> qwen4_batch_prefill(
        qwen4_exp::Model&,std::vector<Qwen4BatchRequestState*>,std::vector<::tyloi::Tensor>);
    friend ::tyloi::Tensor qwen4_batch_decode(
        qwen4_exp::Model&, const std::vector<Qwen4BatchRequestState*>&,
        ::tyloi::Tensor, const ::tyloi::Tensor&);
};
/** Run batch prefill while yielding per-layer resource boundaries. */
tyloi::StepSequence<std::vector<::tyloi::Tensor>> qwen4_batch_prefill(
    qwen4_exp::Model&,std::vector<Qwen4BatchRequestState*>,std::vector<::tyloi::Tensor>);
::tyloi::Tensor qwen4_batch_decode(
    qwen4_exp::Model&, const std::vector<Qwen4BatchRequestState*>&,
    ::tyloi::Tensor ids, const ::tyloi::Tensor& host_ids);
class Qwen4BatchDecodeGraph {
  public:
    struct Plan {
        std::vector<int64_t> capacities;
        std::size_t growth = 0;
        bool rebuild = false;
        std::unique_ptr<tyloi::ops::Qsa::Graph> attention;
    };
    explicit Qwen4BatchDecodeGraph(qwen4_exp::Model&);
    ~Qwen4BatchDecodeGraph();
    /** Plan a graph while yielding required runtime resource boundaries. */
    tyloi::StepSequence<Plan> plan(std::vector<Qwen4BatchRequestState*>, bool greedy);
    std::optional<::tyloi::Tensor> run(
        const std::vector<Qwen4BatchRequestState*>&,
        const ::tyloi::Tensor&,const ::tyloi::Tensor&,bool greedy,const Plan&);
    void clear();
    int64_t captures() const;
    int64_t replays() const;
  private:
    struct Impl;
    std::unique_ptr<Impl> graph_;
};
void qwen4_report_decode_graph(const qwen4_exp::Model&,const qwen4_exp::State&,std::ostream&);
std::vector<std::pair<std::string,double>> qwen4_decode_graph_profile_metrics(const qwen4_exp::Model&,const qwen4_exp::State&);

namespace qwen4_exp {

// The model record owns weights and load-time state.
struct Model : tyloi::models::CausalModel {
    using CausalConfig = tyloi::models::qwen4_exp::Config;
    static bool accepts_backbone(std::string_view backbone) { return backbone == "qwen4_exp"; }

    // ---- tyloi::models::CausalModel
    /** Report this model's dimensions and execution policy. */
    tyloi::models::Info info() const override;
    /** Select the logical span and workspace policy for this request. */
    tyloi::models::PrefillPlan prefill_plan(const tyloi::models::State&, const tyloi::models::PrefillOptions&) const override;
    /** Report capabilities of the components actually loaded. */
    MfqModelCapabilities capabilities() const override;
    /** Report component availability and supported KV memory policy. */
    tyloi::engine::Metrics metrics() const override;
    /** Borrow the predictor owned by this model, when present. */
    tyloi::models::Mtp* mtp() const noexcept override;
    /** Create request state and its predictor state for this model. */
    std::unique_ptr<tyloi::models::State> make_state() const override;
    /** Warm eligible model graphs using this model's request state. */
    tyloi::StepSequence<tyloi::models::Decoded> warmup(tyloi::models::State& state, tyloi::models::DecodeOptions options) const override;
    /** Create model-owned batch computation and optional decode graph state. */
    std::unique_ptr<tyloi::models::Batch> batch(const tyloi::models::BatchConfig& config, tyloi::models::State* serial) override;
    /** Prepare media using the creating model's request-owned cache. */
    tyloi::StepSequence<tyloi::models::Prepared> prepare(const std::vector<std::int64_t>& tokens,
        const MfqMultimodalInput& media, tyloi::models::State& state) const override;
    ::tyloi::Tensor embedding(const ::tyloi::Tensor& ids, int primary = 0) const override;
    ::tyloi::Tensor normalize(const ::tyloi::Tensor& hidden, std::int64_t batch,
        std::int64_t tokens) const override;
    ::tyloi::Tensor softcap(const ::tyloi::Tensor& logits) const override;
    ::tyloi::Tensor logits(const ::tyloi::Tensor& hidden, bool capped = true) const override;
    ::tyloi::Tensor forward(tyloi::models::Input input, tyloi::models::State& state,
                                 const tyloi::models::ForwardOptions& options = {}) const override;
    /** Run an eligible model-owned decode graph and return its requested projection. */
    tyloi::StepSequence<tyloi::models::Decoded> decode(tyloi::models::Input input, tyloi::models::State& state,
        tyloi::models::ForwardOptions options, tyloi::models::DecodeOptions decode) const override;
    /** Run eligible layer-major prefill in resumable model-owned steps. */
    tyloi::StepSequence<tyloi::models::Prefill> prefill(tyloi::models::Input input,
        tyloi::models::State& state, const tyloi::models::ForwardOptions& options,
        tyloi::models::PrefillOptions prefill) const override;

    // ---- speculative contract used by the MTP sequence
    bool supports_speculation() const noexcept { return true; }
    bool supports_suffix_speculation() const noexcept { return true; }
    /** Begin row prefetch for the creating request's upcoming host token IDs. */
    bool prefetch(tyloi::models::State& state, const ::tyloi::Tensor& ids) const override;

    // ---- weights and load-time state
    tyloi::models::CausalLmMetadata metadata;
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    ::tyloi::models::Plan plan;
    ::tyloi::Tensor embed;
    ::tyloi::Tensor lm_head;
    CausalConfig config;
    std::unique_ptr<tyloi::ops::Residual> final_mixer;
    std::vector<std::unique_ptr<qwen4_exp::Layer>> layers;
    /** Primary device selected by the model's load placement. */
    int primary_device = 0;
    /** Retain host token IDs for configured prefetch and CPU layer placement. */
    bool needs_host_tokens = false;
    /** Tensor or expert parallel placement prevents layer-major prefill. */
    bool parallel = false;
    /** Loaded optional weights remain owned by the shared model. */
    std::unique_ptr<Components> components;

    // ---- helpers absorbed from the retired CausalLm plumbing
    ::tyloi::DenseTensor device_ids(::tyloi::DenseTensor ids) const {
        return tyloi::ops::contiguous(tyloi::ops::copy(
            tyloi::ops::cast(ids, ::tyloi::kInt64),
            ::tyloi::Device(::tyloi::kCUDA, primary_device))).dense();
    }
    ::tyloi::DenseTensor embed_tokens(::tyloi::DenseTensor ids) const {
        return tyloi::ops::measure("model.embed", ids.device(), [&] {
            return tyloi::ops::embedding(ids, embed);
        }).dense();
    }
    std::int64_t hidden_size() const noexcept { return metadata.hidden_size; }
    std::int64_t max_position_embeddings() const noexcept { return config.maximum; }
    std::int64_t num_hidden_layers() const noexcept { return config.layers; }
    bool tie_word_embeddings() const noexcept { return metadata.tie_word_embeddings; }
    void load_config(std::string_view payload);
    static ::tyloi::DenseTensor repeat_hidden(::tyloi::DenseTensor hidden, std::int64_t streams) {
        return hidden.to(::tyloi::kFloat16).repeat({1, 1, streams});
    }
    Model();
    ~Model();
    Model(Model&&) noexcept;
    Model& operator=(Model&&) noexcept;

};

void read(Model& model, std::shared_ptr<const tyloi::ModelSource> source,
          std::string_view definition, std::int64_t context_size);
/** Load weights into an already configured model using the selected placement. */
void load(Model& model, const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
          const tyloi::models::LoadOptions& options, bool load_layers = true);
/** Assemble a configured model and its optional components from the supplied source. */
void load(std::unique_ptr<tyloi::models::CausalModel>& owner, std::shared_ptr<const tyloi::ModelSource> source,
    std::string_view definition, const tyloi::weights::Loader& weights,
    LayerPlacementConfig& placement, const tyloi::models::LoadOptions& options);

} // namespace tyloi::cuda::qwen4_exp

using qwen4_exp::Model;
} // namespace tyloi::cuda
