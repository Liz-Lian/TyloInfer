#pragma once

#include "engine/models/common/causal_model_ops.h"
#include "engine/models/causal_model.h"
#include "engine/models/info.h"
#include "engine/metrics.h"
#include "engine/models/common/prompt.h"
#include "engine/models/qwen4_exp/block.h"
#include "engine/models/common/session/size.h"
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
struct MtpModule;

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
    std::size_t prefill_growth_bytes(int64_t tokens) const;
  private:
    struct Impl;
    std::unique_ptr<Impl> state_;
    friend class Qwen4BatchDecodeGraph;
    friend std::vector<::tyloi::DenseTensor> qwen4_batch_prefill(
        qwen4_exp::Model&,const std::vector<Qwen4BatchRequestState*>&,
        const std::vector<::tyloi::DenseTensor>&);
    friend ::tyloi::DenseTensor qwen4_batch_decode(
        qwen4_exp::Model&, const std::vector<Qwen4BatchRequestState*>&,
        ::tyloi::DenseTensor, const ::tyloi::DenseTensor&);
};
std::vector<::tyloi::DenseTensor> qwen4_batch_prefill(
    qwen4_exp::Model&,const std::vector<Qwen4BatchRequestState*>&,
    const std::vector<::tyloi::DenseTensor>&);
::tyloi::DenseTensor qwen4_batch_decode(
    qwen4_exp::Model&, const std::vector<Qwen4BatchRequestState*>&,
    ::tyloi::DenseTensor ids, const ::tyloi::DenseTensor& host_ids);
class Qwen4BatchDecodeGraph {
  public:
    explicit Qwen4BatchDecodeGraph(qwen4_exp::Model&);
    ~Qwen4BatchDecodeGraph();
    std::optional<::tyloi::DenseTensor> run(
        const std::vector<Qwen4BatchRequestState*>&,
        const ::tyloi::DenseTensor&,const ::tyloi::DenseTensor&,bool greedy=false);
    void clear();
    int64_t captures() const;
    int64_t replays() const;
  private:
    struct Impl;
    std::unique_ptr<Impl> graph_;
};
void qwen4_begin_router_lookahead_audit(qwen4_exp::Model&,int samples);
void qwen4_finish_router_lookahead_audit(qwen4_exp::Model&,const std::string& prefix);
void qwen4_set_rotary_fusion(qwen4_exp::Model&, bool enabled);
void qwen4_set_attention_grouping(qwen4_exp::Model&, bool enabled);
bool qwen4_prefetch_decode_ple(qwen4_exp::Model&,const ::tyloi::DenseTensor& host_ids);
std::optional<::tyloi::DenseTensor> qwen4_decode_graph_logits(
    qwen4_exp::Model&,::tyloi::DenseTensor,int,const ::tyloi::DenseTensor& host_ids={});
std::optional<::tyloi::DenseTensor> qwen4_decode_graph_hidden(
    qwen4_exp::Model&,::tyloi::DenseTensor,::tyloi::DenseTensor* raw_hidden,int64_t confirmed=0,
    bool* initialized=nullptr,const ::tyloi::DenseTensor& host_ids={},
    ::tyloi::DenseTensor* logits=nullptr,std::vector<int32_t>* greedy_tokens=nullptr,bool compact_sampling=true);
void qwen4_report_decode_graph(const qwen4_exp::Model&,std::ostream&);
std::vector<std::pair<std::string,double>> qwen4_decode_graph_profile_metrics(const qwen4_exp::Model&,const MtpModule*);
::tyloi::DenseTensor qwen4_decode_graph_compact_topk(const qwen4_exp::Model&);
void qwen4_release_decode_graphs(qwen4_exp::Model&,MtpModule*);
void qwen4_warmup_decode_graphs(qwen4_exp::Model&,MtpModule*,int maximum_draft_depth=3);
std::size_t qwen4_mtp_graph_reserve(const qwen4_exp::Model&,const MtpModule*,int maximum_draft_depth=5);
tyloi::models::SnapshotSize snapshot_size(const qwen4_exp::Model&,const MtpModule*);
std::vector<std::pair<std::string,double>> qwen4_cache_memory_metrics(const qwen4_exp::Model&,const MtpModule*);
bool qwen4_layer_major_prefill_eligible(const qwen4_exp::Model&, int64_t tokens);
std::int64_t prefill_span(const qwen4_exp::Model&, std::size_t tokens,
                          std::int64_t chunk, std::size_t reused,
                          std::size_t stable, bool transformed, bool);
::tyloi::DenseTensor qwen4_layer_major_prefill_hidden(
    qwen4_exp::Model&, ::tyloi::DenseTensor ids, int64_t compute_chunk = 1024,
    bool last_only = true, const std::function<bool()>& cancelled = {},
    ::tyloi::DenseTensor* raw_hidden = nullptr, bool raw_hidden_host = false);
// Report completed prompt computation before restoring lent expert storage.
::tyloi::DenseTensor qwen4_layer_major_prefill_hidden(
    qwen4_exp::Model&, ::tyloi::DenseTensor ids, int64_t compute_chunk,
    bool last_only, const std::function<bool()>& cancelled,
    ::tyloi::DenseTensor* raw_hidden, bool raw_hidden_host,
    const std::function<void()>& computation_finished);

namespace qwen4_exp {

// The model record: weights, decode-graph registry and load-time state.
// Request position and speculative journal live in State; the recurrent
// caches inside Gdn/Qsa/Ple follow them once requests stop being single-flight.
struct Model : CausalResources, tyloi::models::CausalModel {
    using CausalConfig = tyloi::models::qwen4_exp::Config;
    static bool accepts_backbone(std::string_view backbone) { return backbone == "qwen4_exp"; }

    // ---- tyloi::models::CausalModel
    tyloi::models::Info info() const override;
    MfqModelCapabilities capabilities() const override;
    tyloi::engine::Metrics metrics() const override;
    std::unique_ptr<tyloi::models::State> make_state() const override;
    tyloi::StepSequence<tyloi::models::Prepared> prepare(const std::vector<std::int64_t>& tokens,
        const MfqMultimodalInput& media, tyloi::models::State& state) const override;
    ::tyloi::DenseTensor embedding(::tyloi::DenseTensor ids, int primary = 0) const override;
    ::tyloi::DenseTensor normalize(::tyloi::DenseTensor hidden, std::int64_t batch,
        std::int64_t tokens) const override;
    ::tyloi::DenseTensor softcap(::tyloi::DenseTensor logits) const override;
    ::tyloi::DenseTensor logits(::tyloi::DenseTensor hidden, bool capped = true) const override;
    ::tyloi::DenseTensor forward(tyloi::models::Input input, tyloi::models::State& state,
                                 const tyloi::models::ForwardOptions& options = {}) const override;

    // ---- speculative contract used by the MTP sequence
    bool supports_speculation() const noexcept { return true; }
    bool supports_suffix_speculation() const noexcept { return true; }
    ::tyloi::DenseTensor hidden_forward_speculative_suffix(::tyloi::DenseTensor ids,
                                                           ::tyloi::DenseTensor* raw_hidden = nullptr);
    void rollback_speculative(std::int64_t accepted);
    bool adapter_prefetch_token_rows(Model& model, const ::tyloi::DenseTensor& ids);
    std::optional<::tyloi::DenseTensor> adapter_graph_logits(Model& model, ::tyloi::DenseTensor ids,
                                                             int kind);

    // ---- weights and load-time state
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    CudaModelPlan plan;
    RopeCache rope;
    RopeCache cpu_rope;
    std::unordered_map<int, RopeCache> device_ropes;
    std::shared_ptr<const ::tyloi::Tensor> embed;
    ::tyloi::DenseTensor output_norm;
    std::shared_ptr<const ::tyloi::Tensor> lm_head;
    CausalConfig config;
    std::unique_ptr<GatedResidual> final_mixer;
    ::tyloi::DenseTensor positions;
    int64_t batch = 0;
    std::optional<bool> router_lookahead;
    std::unique_ptr<Qwen4DecodeGraph> graph_registry;
    bool reuse_decode_graphs=false;
    uint64_t decode_graph_captures=0;
    // Bounded request lookahead, independent of the configured context ceiling.
    int64_t decode_reserve_until=0;
    bool speculative_prefix_retention=false;
    std::vector<std::unique_ptr<qwen4_exp::Qwen4Block>> blocks;
    std::function<void(std::size_t)> reserve_kv_workspace;
    // shortcut: states are single-flight per engine; the bridge is released in ~State
    State* active = nullptr;

    // ---- helpers absorbed from the retired CausalLm plumbing
    auto execution_scope() const {
        return CudaScope(execution->config.layer_placement.primary_device(), execution->executor().contexts());
    }
    ::tyloi::DenseTensor device_ids(::tyloi::DenseTensor ids) const {
        return execution->executor().collectives()
            .copy(ids.to(::tyloi::kInt64), execution->config.layer_placement.primary_device())
            .contiguous();
    }
    ::tyloi::DenseTensor embed_tokens(::tyloi::DenseTensor ids) const {
        return execution->executor().profiler().measure(
            "model.embed", [&] { return adapter_embed(quant_embedding_lookup(*embed, ids)); });
    }
    std::int64_t hidden_size() const noexcept { return metadata.hidden_size; }
    std::int64_t max_position_embeddings() const noexcept { return config.maximum; }
    std::int64_t num_hidden_layers() const noexcept { return config.layers; }
    bool tie_word_embeddings() const noexcept { return metadata.tie_word_embeddings; }
    template <class Loader> bool has_independent_output_weights(Loader &loader) const {
        return !tie_word_embeddings() && loader.has_weight("model.output.weight");
    }
    template <class Loader> void load_weights(Loader &loader, bool with_blocks = true) {
        constexpr auto embedding = "model.token_embedding.weight";
        constexpr auto output = "model.output.weight";
        this->embed = loader.embedding(embedding);
        loader.final_state(this->output_norm);
        this->lm_head = tie_word_embeddings() || !loader.has_weight(output)
                            ? loader.tied_output(this->embed, embedding)
                            : loader.output(output);
        if (with_blocks) {
            loader.prepare_blocks();
            this->blocks.reserve(static_cast<std::size_t>(num_hidden_layers()));
            for (int layer = 0; layer < num_hidden_layers(); ++layer)
                this->blocks.push_back(loader.block(layer, std::string(metadata.layer_types.at(static_cast<std::size_t>(layer)))));
        }
    }
    void load_config(std::string_view payload);
    bool adapter_requires_batch_reset(std::int64_t new_batch) const noexcept {
        return batch != 0 && batch != new_batch;
    }
    static ::tyloi::DenseTensor repeat_hidden(::tyloi::DenseTensor hidden, std::int64_t streams) {
        return hidden.to(::tyloi::kFloat16).repeat({1, 1, streams});
    }
    ::tyloi::DenseTensor adapter_prepare_hidden(::tyloi::DenseTensor hidden, std::int64_t,
                                                std::int64_t) const {
        return repeat_hidden(std::move(hidden), metadata.hc_mult);
    }
    bool adapter_force_cache_advance() const noexcept { return true; }
    void adapter_validate_forward(std::int64_t, std::int64_t tokens, std::int64_t cache_position,
                                  bool, bool has_cache_position_override, bool has_attention_mask) const {
        tyloi::models::require_model(tokens > 0 &&
                              cache_position + tokens <= metadata.max_position_embeddings &&
                                  !has_cache_position_override && !has_attention_mask,
                              "Qwen4 requires unpadded causal cache positions");
    }
    void adapter_validate_positions(const ::tyloi::DenseTensor& positions, std::int64_t batch,
                                    std::int64_t tokens, bool) const {
        tyloi::models::require_model((positions.dim() == 1 || (positions.dim() == 2 && positions.size(0) == 3) ||
                                 (positions.dim() == 3 && positions.size(0) == 3 && positions.size(1) == batch)) &&
                              positions.size(-1) == tokens,
                              "Qwen4 positions require [T], [3,T], [4,T], [3,B,T] or [4,B,T]");
    }
    ::tyloi::DenseTensor adapter_prepare_positions(::tyloi::DenseTensor current, std::int64_t,
                                                   std::int64_t) {
        if ((current.dim() == 2 || current.dim() == 3) && current.size(0) == 4)
            current = current.narrow(0, 1, 3);
        return current;
    }
    bool adapter_allows_speculative_position_override() const noexcept { return true; }
    ::tyloi::DenseTensor
    adapter_finalize_hidden(::tyloi::DenseTensor hidden, const ::tyloi::DenseTensor &output_norm,
                            std::int64_t batch, std::int64_t tokens) const;

    // ---- family adapters (implementations in ops.cpp)
    Model();
    ~Model();
    Model(Model&&) noexcept;
    Model& operator=(Model&&) noexcept;
    void adapter_validate_load_options() const;
    void adapter_load_final_state(const tyloi::cuda::weight_loader::Loader& weights,
                                  ::tyloi::DenseTensor& output_norm);
    std::unique_ptr<qwen4_exp::Qwen4Block> adapter_load_block(
        const tyloi::cuda::weight_loader::Loader& weights, int layer, int device,
        const std::string& type);
    void adapter_reset(int64_t batch);
    void adapter_begin_forward(bool capture_raw_hidden);
    ::tyloi::DenseTensor
    adapter_block_positions(const ::tyloi::DenseTensor& full_positions,
                            const ::tyloi::DenseTensor& local_positions, int device) const;
    void adapter_finish_forward(const ::tyloi::DenseTensor& full_positions, int64_t batch,
                                int64_t tokens);
    ::tyloi::DenseTensor adapter_last_logits(const ::tyloi::Tensor& lm_head,
                                             ::tyloi::DenseTensor hidden) const;
    ::tyloi::DenseTensor adapter_next_token(const ::tyloi::Tensor& lm_head,
                                            ::tyloi::DenseTensor hidden) const;
    void adapter_rollback_speculative(int64_t keep);
    void adapter_commit_speculative() { speculative_prefix_retention=false; }
};

void read(Model& model, std::shared_ptr<const tyloi::ModelSource> source,
          std::string_view definition, std::int64_t context_size);
tyloi::models::qwen4_exp::BlockSnapshot capture_block_session(const Qwen4Block& block,
                                                              std::int64_t position, std::size_t& bytes);
void restore_block_session(Qwen4Block& block, const tyloi::models::qwen4_exp::BlockSnapshot& state,
                           std::int64_t position);
void load(Model& model, CudaExecutionContext& execution, std::int64_t context_size,
          bool defer_moe_cache_finalize, bool load_blocks);

} // namespace tyloi::cuda::qwen4_exp

using qwen4_exp::Model;
} // namespace tyloi::cuda
