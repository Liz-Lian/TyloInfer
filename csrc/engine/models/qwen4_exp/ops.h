#pragma once

#include "engine/ops.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/ops/qsa.h"
#endif
#include "engine/models/causal_model.h"
#include "engine/models/batch.h"
#include "engine/models/metadata.h"
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

#ifndef TYLOINFER_NATIVE_CUDA
struct CudaExecutionContext;
#endif
namespace tyloi::cuda::qwen4_exp { struct Layer; struct Components; }
#ifndef TYLOINFER_NATIVE_CUDA
namespace tyloi::cuda { class GatedResidual; }
#endif
namespace tyloi::models { struct Mtp; struct LoadOptions; }
namespace tyloi::weights { struct Loader; }
struct LayerPlacementConfig;

namespace tyloi::cuda {
struct Qwen4DecodeGraph;
class Qwen4BatchDecodeGraph;
// A request owns recurrent, positional and attention state; weights stay shared.
class Qwen4BatchRequestState
#ifdef TYLOINFER_NATIVE_CUDA
    : public tyloi::models::BatchState
#endif
{
  public:
    explicit Qwen4BatchRequestState(qwen4_exp::Model&);
#ifdef TYLOINFER_NATIVE_CUDA
    ~Qwen4BatchRequestState() override;
#else
    ~Qwen4BatchRequestState();
#endif
    Qwen4BatchRequestState(const Qwen4BatchRequestState&) = delete;
    Qwen4BatchRequestState& operator=(const Qwen4BatchRequestState&) = delete;
#ifdef TYLOINFER_NATIVE_CUDA
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
#else
    int64_t position() const;
    std::size_t resident_bytes() const;
    std::size_t initial_growth_bytes() const;
    void initialize();
    std::size_t prefill_growth_bytes(int64_t tokens) const;
#endif
  private:
    struct Impl;
    std::unique_ptr<Impl> state_;
    friend class Qwen4BatchDecodeGraph;
#ifdef TYLOINFER_NATIVE_CUDA
    friend tyloi::StepSequence<std::vector<::tyloi::Tensor>> qwen4_batch_prefill(
        qwen4_exp::Model&,std::vector<Qwen4BatchRequestState*>,std::vector<::tyloi::Tensor>);
#else
    friend std::vector<::tyloi::Tensor> qwen4_batch_prefill(
        qwen4_exp::Model&,const std::vector<Qwen4BatchRequestState*>&,
        const std::vector<::tyloi::Tensor>&);
#endif
    friend ::tyloi::Tensor qwen4_batch_decode(
        qwen4_exp::Model&, const std::vector<Qwen4BatchRequestState*>&,
        ::tyloi::Tensor, const ::tyloi::Tensor&);
};
#ifdef TYLOINFER_NATIVE_CUDA
/** Run batch prefill while yielding per-layer resource boundaries. */
tyloi::StepSequence<std::vector<::tyloi::Tensor>> qwen4_batch_prefill(
    qwen4_exp::Model&,std::vector<Qwen4BatchRequestState*>,std::vector<::tyloi::Tensor>);
#else
std::vector<::tyloi::Tensor> qwen4_batch_prefill(
    qwen4_exp::Model&,const std::vector<Qwen4BatchRequestState*>&,
    const std::vector<::tyloi::Tensor>&);
#endif
::tyloi::Tensor qwen4_batch_decode(
    qwen4_exp::Model&, const std::vector<Qwen4BatchRequestState*>&,
    ::tyloi::Tensor ids, const ::tyloi::Tensor& host_ids);
class Qwen4BatchDecodeGraph {
  public:
    struct Plan {
        std::vector<int64_t> capacities;
        std::size_t growth = 0;
        bool rebuild = false;
#ifdef TYLOINFER_NATIVE_CUDA
        std::unique_ptr<tyloi::ops::Qsa::Graph> attention;
#endif
    };
    explicit Qwen4BatchDecodeGraph(qwen4_exp::Model&);
    ~Qwen4BatchDecodeGraph();
#ifdef TYLOINFER_NATIVE_CUDA
    /** Plan a graph while yielding required runtime resource boundaries. */
    tyloi::StepSequence<Plan> plan(std::vector<Qwen4BatchRequestState*>, bool greedy);
#else
    std::optional<Plan> plan(const std::vector<Qwen4BatchRequestState*>&, bool greedy);
#endif
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
#ifndef TYLOINFER_NATIVE_CUDA
std::optional<::tyloi::Tensor> qwen4_decode_graph_logits(
    const qwen4_exp::Model&,qwen4_exp::State&,::tyloi::Tensor,int,const ::tyloi::Tensor& host_ids={});
std::optional<::tyloi::Tensor> qwen4_decode_graph_hidden(
    const qwen4_exp::Model&,qwen4_exp::State&,::tyloi::Tensor,::tyloi::Tensor* raw_hidden,int64_t confirmed=0,
    bool* initialized=nullptr,const ::tyloi::Tensor& host_ids={},
    ::tyloi::Tensor* logits=nullptr,std::vector<int32_t>* greedy_tokens=nullptr,bool compact_sampling=true);
::tyloi::Tensor qwen4_decode_graph_compact_topk(const qwen4_exp::Model&,const qwen4_exp::State&);
#endif
void qwen4_report_decode_graph(const qwen4_exp::Model&,const qwen4_exp::State&,std::ostream&);
std::vector<std::pair<std::string,double>> qwen4_decode_graph_profile_metrics(const qwen4_exp::Model&,const qwen4_exp::State&);
#ifndef TYLOINFER_NATIVE_CUDA
void qwen4_release_decode_graphs(const qwen4_exp::Model&,qwen4_exp::State&);
void qwen4_warmup_decode_graphs(const qwen4_exp::Model&,qwen4_exp::State&);
#endif
#ifndef TYLOINFER_NATIVE_CUDA
std::vector<std::pair<std::string,double>> qwen4_cache_memory_metrics(const qwen4_exp::Model&,const qwen4_exp::State&,const tyloi::models::Mtp*);
#endif
#ifndef TYLOINFER_NATIVE_CUDA
bool qwen4_layer_major_prefill_eligible(const qwen4_exp::Model&, const qwen4_exp::State&, int64_t tokens);
std::int64_t prefill_span(const qwen4_exp::Model&, const qwen4_exp::State&, std::size_t tokens,
                          std::int64_t chunk, std::size_t reused,
                          std::size_t stable, bool transformed, bool);
#endif
#ifndef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor qwen4_layer_major_prefill_hidden(
    qwen4_exp::Model&, qwen4_exp::State&, ::tyloi::Tensor ids, int64_t compute_chunk = 1024,
    bool last_only = true, const std::function<bool()>& cancelled = {},
    ::tyloi::Tensor* raw_hidden = nullptr, bool raw_hidden_host = false);
// Report completed prompt computation before restoring lent expert storage.
::tyloi::Tensor qwen4_layer_major_prefill_hidden(
    qwen4_exp::Model&, qwen4_exp::State&, ::tyloi::Tensor ids, int64_t compute_chunk,
    bool last_only, const std::function<bool()>& cancelled,
    ::tyloi::Tensor* raw_hidden, bool raw_hidden_host,
    const std::function<void()>& computation_finished);
#endif

namespace qwen4_exp {

// The model record owns weights and load-time state.
struct Model : tyloi::models::CausalModel {
    using CausalConfig = tyloi::models::qwen4_exp::Config;
    static bool accepts_backbone(std::string_view backbone) { return backbone == "qwen4_exp"; }

    // ---- tyloi::models::CausalModel
    /** Report this model's dimensions and execution policy. */
    tyloi::models::Info info() const override;
#ifdef TYLOINFER_NATIVE_CUDA
    /** Select the logical span and workspace policy for this request. */
    tyloi::models::PrefillPlan prefill_plan(const tyloi::models::State&, const tyloi::models::PrefillOptions&) const override;
#endif
    /** Report capabilities of the components actually loaded. */
    MfqModelCapabilities capabilities() const override;
    /** Report component availability and supported KV memory policy. */
    tyloi::engine::Metrics metrics() const override;
    /** Borrow the predictor owned by this model, when present. */
    tyloi::models::Mtp* mtp() const noexcept override;
    /** Create request state and its predictor state for this model. */
    std::unique_ptr<tyloi::models::State> make_state() const override;
#ifdef TYLOINFER_NATIVE_CUDA
    /** Warm eligible model graphs using this model's request state. */
    tyloi::StepSequence<tyloi::models::Decoded> warmup(tyloi::models::State& state, tyloi::models::DecodeOptions options) const override;
    /** Create model-owned batch computation and optional decode graph state. */
    std::unique_ptr<tyloi::models::Batch> batch(const tyloi::models::BatchConfig& config, tyloi::models::State* serial) override;
#endif
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
#ifdef TYLOINFER_NATIVE_CUDA
    /** Run an eligible model-owned decode graph and return its requested projection. */
    tyloi::StepSequence<tyloi::models::Decoded> decode(tyloi::models::Input input, tyloi::models::State& state,
        tyloi::models::ForwardOptions options, tyloi::models::DecodeOptions decode) const override;
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    /** Run eligible layer-major prefill in resumable model-owned steps. */
    tyloi::StepSequence<tyloi::models::Prefill> prefill(tyloi::models::Input input,
        tyloi::models::State& state, const tyloi::models::ForwardOptions& options,
        tyloi::models::PrefillOptions prefill) const override;
#endif

    // ---- speculative contract used by the MTP sequence
    bool supports_speculation() const noexcept { return true; }
    bool supports_suffix_speculation() const noexcept { return true; }
#ifndef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor hidden_forward_speculative_suffix(State& state, ::tyloi::Tensor ids,
                                                      ::tyloi::Tensor* raw_hidden = nullptr);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    /** Begin row prefetch for the creating request's upcoming host token IDs. */
    bool prefetch(tyloi::models::State& state, const ::tyloi::Tensor& ids) const override;
#else
    bool prefetch_token_rows(State& state, const ::tyloi::Tensor& ids);
#endif

    // ---- weights and load-time state
    tyloi::models::CausalLmMetadata metadata;
#ifndef TYLOINFER_NATIVE_CUDA
    CudaExecutionContext* execution = nullptr;
#endif
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    CudaModelPlan plan;
    ::tyloi::Tensor embed;
    ::tyloi::Tensor lm_head;
    CausalConfig config;
#ifdef TYLOINFER_NATIVE_CUDA
    std::unique_ptr<tyloi::ops::Residual> final_mixer;
#else
    std::unique_ptr<GatedResidual> final_mixer;
#endif
    std::vector<std::unique_ptr<qwen4_exp::Layer>> layers;
#ifdef TYLOINFER_NATIVE_CUDA
    /** Primary device selected by the model's load placement. */
    int primary_device = 0;
    /** Retain host token IDs for configured prefetch and CPU layer placement. */
    bool needs_host_tokens = false;
    /** Tensor or expert parallel placement prevents layer-major prefill. */
    bool parallel = false;
    /** Loaded optional weights remain owned by the shared model. */
    std::unique_ptr<Components> components;
#endif

    // ---- helpers absorbed from the retired CausalLm plumbing
#ifndef TYLOINFER_NATIVE_CUDA
    auto execution_scope() const {
        return CudaScope(execution->config.layer_placement.primary_device(), execution->executor().contexts());
    }
#endif
    ::tyloi::DenseTensor device_ids(::tyloi::DenseTensor ids) const {
#ifdef TYLOINFER_NATIVE_CUDA
        return tyloi::ops::contiguous(tyloi::ops::copy(
            tyloi::ops::cast(ids, ::tyloi::kInt64),
            ::tyloi::Device(::tyloi::kCUDA, primary_device))).dense();
#else
        return execution->executor().collectives()
            .copy(ids.to(::tyloi::kInt64), execution->config.layer_placement.primary_device())
            .contiguous();
#endif
    }
    ::tyloi::DenseTensor embed_tokens(::tyloi::DenseTensor ids) const {
#ifdef TYLOINFER_NATIVE_CUDA
        return tyloi::ops::measure("model.embed", ids.device(), [&] {
            return tyloi::ops::embedding(ids, embed);
        }).dense();
#else
        return execution->executor().profiler().measure(
            "model.embed", [&] { return quant_embedding_lookup(embed, ids); });
#endif
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
#ifdef TYLOINFER_NATIVE_CUDA
/** Load weights into an already configured model using the selected placement. */
void load(Model& model, const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
          const tyloi::models::LoadOptions& options, bool load_layers = true);
/** Assemble a configured model and its optional components from the supplied source. */
std::unique_ptr<tyloi::models::CausalModel> load(std::shared_ptr<const tyloi::ModelSource> source,
    std::string_view definition, const tyloi::weights::Loader& weights,
    LayerPlacementConfig& placement, const tyloi::models::LoadOptions& options);
#else
void load(Model& model, CudaExecutionContext& execution, std::int64_t context_size,
          bool defer_moe_cache_finalize, bool load_layers);
#endif

} // namespace tyloi::cuda::qwen4_exp

using qwen4_exp::Model;
} // namespace tyloi::cuda
