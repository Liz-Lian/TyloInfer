#pragma once

#include "engine/ops.h"
#include "engine/models/causal_model.h"
#include "engine/models/metadata.h"
#include "engine/metrics.h"
#include "engine/models/common/prompt.h"
#include "engine/models/qwen4_exp/block.h"
#include "engine/models/session/size.h"
#include "engine/models/qwen4_exp/state.h"
#include "engine/models/qwen4_exp/config.h"
#include "engine/backends/cuda/ops/projection/quant_linear.h"

#include <memory>
#include <functional>
#include <cstdint>
#include <iosfwd>
#include <vector>

namespace tyloi::cuda::qwen4_exp { struct Qwen4Block; }
namespace tyloi::cuda { class GatedResidual; }
namespace tyloi::models { struct Mtp; }

namespace tyloi::cuda {
struct Qwen4DecodeGraph;
class Qwen4BatchDecodeGraph;
// A request owns recurrent, positional and attention state; weights stay shared.
class Qwen4BatchRequestState {
  public:
    explicit Qwen4BatchRequestState(qwen4_exp::Model&);
    ~Qwen4BatchRequestState();
    Qwen4BatchRequestState(const Qwen4BatchRequestState&) = delete;
    Qwen4BatchRequestState& operator=(const Qwen4BatchRequestState&) = delete;
    int64_t position() const;
    std::size_t resident_bytes() const;
    std::size_t initial_growth_bytes() const;
    void initialize();
    std::size_t prefill_growth_bytes(int64_t tokens) const;
  private:
    struct Impl;
    std::unique_ptr<Impl> state_;
    friend class Qwen4BatchDecodeGraph;
    friend std::vector<::tyloi::Tensor> qwen4_batch_prefill(
        qwen4_exp::Model&,const std::vector<Qwen4BatchRequestState*>&,
        const std::vector<::tyloi::Tensor>&);
    friend ::tyloi::Tensor qwen4_batch_decode(
        qwen4_exp::Model&, const std::vector<Qwen4BatchRequestState*>&,
        ::tyloi::Tensor, const ::tyloi::Tensor&);
};
std::vector<::tyloi::Tensor> qwen4_batch_prefill(
    qwen4_exp::Model&,const std::vector<Qwen4BatchRequestState*>&,
    const std::vector<::tyloi::Tensor>&);
::tyloi::Tensor qwen4_batch_decode(
    qwen4_exp::Model&, const std::vector<Qwen4BatchRequestState*>&,
    ::tyloi::Tensor ids, const ::tyloi::Tensor& host_ids);
class Qwen4BatchDecodeGraph {
  public:
    struct Plan {
        std::vector<int64_t> capacities;
        std::size_t growth = 0;
        bool rebuild = false;
    };
    explicit Qwen4BatchDecodeGraph(qwen4_exp::Model&);
    ~Qwen4BatchDecodeGraph();
    std::optional<Plan> plan(const std::vector<Qwen4BatchRequestState*>&, bool greedy);
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
void qwen4_begin_router_lookahead_audit(qwen4_exp::Model&,qwen4_exp::State&,int samples);
void qwen4_finish_router_lookahead_audit(qwen4_exp::Model&,qwen4_exp::State&,const std::string& prefix);
std::optional<::tyloi::Tensor> qwen4_decode_graph_logits(
    qwen4_exp::Model&,qwen4_exp::State&,::tyloi::Tensor,int,const ::tyloi::Tensor& host_ids={});
std::optional<::tyloi::Tensor> qwen4_decode_graph_hidden(
    qwen4_exp::Model&,qwen4_exp::State&,::tyloi::Tensor,::tyloi::Tensor* raw_hidden,int64_t confirmed=0,
    bool* initialized=nullptr,const ::tyloi::Tensor& host_ids={},
    ::tyloi::Tensor* logits=nullptr,std::vector<int32_t>* greedy_tokens=nullptr,bool compact_sampling=true);
void qwen4_report_decode_graph(const qwen4_exp::Model&,const qwen4_exp::State&,std::ostream&);
std::vector<std::pair<std::string,double>> qwen4_decode_graph_profile_metrics(const qwen4_exp::Model&,const qwen4_exp::State&,const tyloi::models::Mtp*);
::tyloi::Tensor qwen4_decode_graph_compact_topk(const qwen4_exp::Model&,const qwen4_exp::State&);
void qwen4_release_decode_graphs(qwen4_exp::Model&,qwen4_exp::State&,tyloi::models::Mtp*);
void qwen4_warmup_decode_graphs(qwen4_exp::Model&,qwen4_exp::State&,tyloi::models::Mtp*,int maximum_draft_depth=3);
std::size_t qwen4_mtp_graph_reserve(const qwen4_exp::Model&,const tyloi::models::Mtp*,int maximum_draft_depth=5);
std::vector<std::pair<std::string,double>> qwen4_cache_memory_metrics(const qwen4_exp::Model&,const qwen4_exp::State&,const tyloi::models::Mtp*);
bool qwen4_layer_major_prefill_eligible(const qwen4_exp::Model&, const qwen4_exp::State&, int64_t tokens);
std::int64_t prefill_span(const qwen4_exp::Model&, const qwen4_exp::State&, std::size_t tokens,
                          std::int64_t chunk, std::size_t reused,
                          std::size_t stable, bool transformed, bool);
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

namespace qwen4_exp {

// The model record owns weights and load-time state.
struct Model : tyloi::models::CausalModel {
    using CausalConfig = tyloi::models::qwen4_exp::Config;
    static bool accepts_backbone(std::string_view backbone) { return backbone == "qwen4_exp"; }

    // ---- tyloi::models::CausalModel
    tyloi::models::Info info() const override;
    MfqModelCapabilities capabilities() const override;
    tyloi::engine::Metrics metrics() const override;
    std::unique_ptr<tyloi::models::State> make_state() const override;
    tyloi::StepSequence<tyloi::models::Prepared> prepare(const std::vector<std::int64_t>& tokens,
        const MfqMultimodalInput& media, tyloi::models::State& state) const override;
    std::shared_ptr<::tyloi::Tensor> embedding(const ::tyloi::Tensor& ids, int primary = 0) const override;
    std::shared_ptr<::tyloi::Tensor> normalize(const ::tyloi::Tensor& hidden, std::int64_t batch,
        std::int64_t tokens) const override;
    std::shared_ptr<::tyloi::Tensor> softcap(const ::tyloi::Tensor& logits) const override;
    std::shared_ptr<::tyloi::Tensor> logits(const ::tyloi::Tensor& hidden, bool capped = true) const override;
    std::shared_ptr<::tyloi::Tensor> forward(tyloi::models::Input input, tyloi::models::State& state,
                                 const tyloi::models::ForwardOptions& options = {}) const override;

    // ---- speculative contract used by the MTP sequence
    bool supports_speculation() const noexcept { return true; }
    bool supports_suffix_speculation() const noexcept { return true; }
    ::tyloi::Tensor hidden_forward_speculative_suffix(State& state, ::tyloi::Tensor ids,
                                                      ::tyloi::Tensor* raw_hidden = nullptr);
    bool prefetch_token_rows(State& state, const ::tyloi::Tensor& ids);

    // ---- weights and load-time state
    tyloi::models::CausalLmMetadata metadata;
    CudaExecutionContext* execution = nullptr;
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    CudaModelPlan plan;
    std::shared_ptr<const ::tyloi::Tensor> embed;
    std::shared_ptr<const ::tyloi::Tensor> lm_head;
    CausalConfig config;
    std::unique_ptr<GatedResidual> final_mixer;
    std::vector<std::unique_ptr<qwen4_exp::Qwen4Block>> blocks;

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
            ::tyloi::Device(::tyloi::kCUDA, execution->config.layer_placement.primary_device()))).dense();
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
tyloi::models::qwen4_exp::BlockSnapshot capture_block_session(const Qwen4Block& block, const LayerState& layer,
                                                              std::int64_t position, std::size_t& bytes);
void restore_block_session(Qwen4Block& block, LayerState& layer, const tyloi::models::qwen4_exp::BlockSnapshot& state,
                           std::int64_t position);
void load(Model& model, CudaExecutionContext& execution, std::int64_t context_size,
          bool defer_moe_cache_finalize, bool load_blocks);

} // namespace tyloi::cuda::qwen4_exp

using qwen4_exp::Model;
} // namespace tyloi::cuda
