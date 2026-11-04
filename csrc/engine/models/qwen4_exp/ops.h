#pragma once

#include "engine/models/common/causal_model_ops.h"
#include "engine/models/common/block.h"
#include "engine/models/common/session/size.h"
#include "engine/models/qwen4_exp/causal_lm.h"
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
    explicit Qwen4BatchRequestState(Qwen4CausalLm&);
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
        Qwen4CausalLm&,const std::vector<Qwen4BatchRequestState*>&,
        const std::vector<::tyloi::DenseTensor>&);
    friend ::tyloi::DenseTensor qwen4_batch_decode(
        Qwen4CausalLm&, const std::vector<Qwen4BatchRequestState*>&,
        ::tyloi::DenseTensor, const ::tyloi::DenseTensor&);
};
std::vector<::tyloi::DenseTensor> qwen4_batch_prefill(
    Qwen4CausalLm&,const std::vector<Qwen4BatchRequestState*>&,
    const std::vector<::tyloi::DenseTensor>&);
::tyloi::DenseTensor qwen4_batch_decode(
    Qwen4CausalLm&, const std::vector<Qwen4BatchRequestState*>&,
    ::tyloi::DenseTensor ids, const ::tyloi::DenseTensor& host_ids);
class Qwen4BatchDecodeGraph {
  public:
    explicit Qwen4BatchDecodeGraph(Qwen4CausalLm&);
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
void qwen4_begin_router_lookahead_audit(Qwen4CausalLm&,int samples);
void qwen4_finish_router_lookahead_audit(Qwen4CausalLm&,const std::string& prefix);
void qwen4_set_rotary_fusion(Qwen4CausalLm&, bool enabled);
void qwen4_set_attention_grouping(Qwen4CausalLm&, bool enabled);
bool qwen4_prefetch_decode_ple(Qwen4CausalLm&,const ::tyloi::DenseTensor& host_ids);
std::optional<::tyloi::DenseTensor> qwen4_decode_graph_logits(
    Qwen4CausalLm&,::tyloi::DenseTensor,int,const ::tyloi::DenseTensor& host_ids={});
std::optional<::tyloi::DenseTensor> qwen4_decode_graph_hidden(
    Qwen4CausalLm&,::tyloi::DenseTensor,::tyloi::DenseTensor* raw_hidden,int64_t confirmed=0,
    bool* initialized=nullptr,const ::tyloi::DenseTensor& host_ids={},
    ::tyloi::DenseTensor* logits=nullptr,std::vector<int32_t>* greedy_tokens=nullptr,bool compact_sampling=true);
void qwen4_report_decode_graph(const Qwen4CausalLm&,std::ostream&);
std::vector<std::pair<std::string,double>> qwen4_decode_graph_profile_metrics(const Qwen4CausalLm&,const MtpModule*);
::tyloi::DenseTensor qwen4_decode_graph_compact_topk(const Qwen4CausalLm&);
void qwen4_release_decode_graphs(Qwen4CausalLm&,MtpModule*);
void qwen4_warmup_decode_graphs(Qwen4CausalLm&,MtpModule*,int maximum_draft_depth=3);
std::size_t qwen4_mtp_graph_reserve(const Qwen4CausalLm&,const MtpModule*,int maximum_draft_depth=5);
tyloi::models::SnapshotSize snapshot_size(const Qwen4CausalLm&,const MtpModule*);
std::vector<std::pair<std::string,double>> qwen4_cache_memory_metrics(const Qwen4CausalLm&,const MtpModule*);
bool qwen4_layer_major_prefill_eligible(const Qwen4CausalLm&, int64_t tokens);
std::int64_t prefill_span(const Qwen4CausalLm&, std::size_t tokens,
                          std::int64_t chunk, std::size_t reused,
                          std::size_t stable, bool transformed, bool);
::tyloi::DenseTensor qwen4_layer_major_prefill_hidden(
    Qwen4CausalLm&, ::tyloi::DenseTensor ids, int64_t compute_chunk = 1024,
    bool last_only = true, const std::function<bool()>& cancelled = {},
    ::tyloi::DenseTensor* raw_hidden = nullptr, bool raw_hidden_host = false);
// Report completed prompt computation before restoring lent expert storage.
::tyloi::DenseTensor qwen4_layer_major_prefill_hidden(
    Qwen4CausalLm&, ::tyloi::DenseTensor ids, int64_t compute_chunk,
    bool last_only, const std::function<bool()>& cancelled,
    ::tyloi::DenseTensor* raw_hidden, bool raw_hidden_host,
    const std::function<void()>& computation_finished);

struct Qwen4Model : CausalResources {
    bool adapter_prefetch_token_rows(Qwen4CausalLm&,const ::tyloi::DenseTensor&);
    Qwen4Model();
    ~Qwen4Model();
    Qwen4Model(Qwen4Model&&) noexcept;
    Qwen4Model& operator=(Qwen4Model&&) noexcept;
    template <class Backend> using CausalModel = tyloi::models::qwen4_exp::CausalLm<Backend>;
    tyloi::models::qwen4_exp::Config config;
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
    std::vector<qwen4_exp::Qwen4Block*> loaded_blocks;
    std::function<void(std::size_t)> reserve_kv_workspace;
    template<class Model> std::optional<::tyloi::DenseTensor>
    adapter_graph_logits(Model& model,::tyloi::DenseTensor ids,int kind) {
        if (ids.dim()==2 && ids.size(0)==1 &&
            qwen4_layer_major_prefill_eligible(model,ids.size(1))) {
            auto hidden=qwen4_layer_major_prefill_hidden(model,std::move(ids),1024,kind!=0);
            if(kind==2)return model.adapter_next_token(model.lm_head,model.last_hidden(std::move(hidden)));
            if(kind==1)return model.apply_final_logit_softcap(
                model.adapter_last_logits(model.lm_head,model.last_hidden(std::move(hidden))));
            return model.logits_from_hidden(std::move(hidden));
        }
        return qwen4_decode_graph_logits(model,std::move(ids),kind);
    }

    void adapter_validate_load_options() const;
    void adapter_load_final_state(const tyloi::cuda::weight_loader::Loader& weights,
                                  ::tyloi::DenseTensor &output_norm);
    std::unique_ptr<Block> adapter_load_block(const tyloi::cuda::weight_loader::Loader& weights, int layer, int device,
                                              const std::string &type);

    void adapter_reset(int64_t batch);
    void adapter_begin_forward(bool capture_raw_hidden);

    ::tyloi::DenseTensor
    adapter_block_positions(const ::tyloi::DenseTensor &full_positions,
                            const ::tyloi::DenseTensor &local_positions, int device) const;
    void adapter_finish_forward(const ::tyloi::DenseTensor &full_positions, int64_t batch,
                                int64_t tokens);
    ::tyloi::DenseTensor
    adapter_finalize_hidden(::tyloi::DenseTensor hidden,
                            const ::tyloi::DenseTensor &output_norm, int64_t batch,
                            int64_t tokens) const;
    ::tyloi::DenseTensor adapter_last_logits(const QuantLinear &lm_head,
                                                   ::tyloi::DenseTensor hidden) const;
    ::tyloi::DenseTensor adapter_next_token(const QuantLinear &lm_head,
                                                  ::tyloi::DenseTensor hidden) const;
    void adapter_rollback_speculative(int64_t keep);
    void adapter_commit_speculative() { speculative_prefix_retention=false; }
};

template <> struct CudaSessionCodec<Qwen4Model> {
    static TextSessionStateKind kind(const Qwen4CausalLm&);
    static bool supports_paged(const Qwen4CausalLm&);
    static TextSessionState capture(const Qwen4CausalLm&,const std::vector<int64_t>&);
    static void restore(Qwen4CausalLm&,const TextSessionState&);
};

} // namespace tyloi::cuda

namespace tyloi::models {
extern template struct qwen4_exp::CausalLm<cuda::CudaCausalOps<cuda::Qwen4Model>>;
} // namespace tyloi::models
