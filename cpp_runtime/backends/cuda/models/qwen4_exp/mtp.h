#pragma once

#include "core/mtp.h"
#include "core/attention.h"
#include "models/qwen4_exp/config.h"

namespace mfq::cuda::qwen4_exp {
using attention_ops::Linear;
struct Gr;
struct Qwen4Block;
struct Qwen4MtpGraph;

struct Qwen4ExpMtp final : MtpModule {
    using Tensor = mfq_tensor_backend::Tensor;

    mfq::models::qwen4_exp::Config config;
    Tensor embedding_norm, hidden_norm;
    Linear embedding_fusion, hidden_fusion;
    Linear draft_output;
    Tensor draft_token_ids;
    std::unique_ptr<Gr> final_mixer;
    std::vector<std::unique_ptr<Qwen4Block>> layers;
    std::vector<Tensor> positions;
    std::vector<int64_t> lengths;
    CudaExecutionContext *execution = nullptr;
    int64_t batch = 0;
    std::unique_ptr<Qwen4MtpGraph> graph_registry;
    bool reuse_decode_graphs=false;
    uint64_t graph_captures=0;
    uint64_t graph_replays = 0;
    uint64_t history_tail_replays = 0;
    int64_t dense_attention_window = -1;
    bool graph_initialized = false;
    bool compact_graph_sampling = true;
    Tensor graph_compact_topk() const;
    bool take_graph_initialization() noexcept override { return std::exchange(graph_initialized,false); }
    uint64_t graph_replay_count() const noexcept override { return graph_replays; }
    std::optional<std::size_t> extra_decode_workspace_bytes(int) const override;

    Qwen4ExpMtp();
    ~Qwen4ExpMtp() override;
    Qwen4ExpMtp(Qwen4ExpMtp&&) noexcept;
    Qwen4ExpMtp& operator=(Qwen4ExpMtp&&) noexcept;

    static std::optional<Qwen4ExpMtp> load_if_present(CudaExecutionContext &execution,
                                                      const mfq::ModelSource &file,
                                                      const mfq::models::qwen4_exp::Config &main);

    void reset(int64_t next_batch = 1) override;

    std::pair<Tensor, Tensor> evaluate(const MtpTarget &target, const Tensor &hidden,
                                       const Tensor &ids, int64_t depth = 0, bool cache = true,
                                       const Tensor &supplied_positions = {},
                                       const Tensor &supplied_embeddings = {},
                                       bool history_only = false);

    Tensor forward(const MtpTarget &target, Tensor hidden, Tensor ids) override;
    Tensor logits_for(const MtpTarget &target, Tensor hidden) const;
    bool supports_graph_window(int64_t tokens) const;

    MtpStep step(const MtpTarget &target, Tensor hidden, Tensor ids) override;
    MtpStep step_positioned(const MtpTarget &target, Tensor hidden, Tensor ids, Tensor positions) override;
    void history_step(const MtpTarget &target, Tensor hidden, Tensor ids, Tensor positions = {},
                      bool scalar_chain = false);
    std::optional<MtpStep> history_tail_step(const MtpTarget&,Tensor hidden,Tensor ids,Tensor positions = {});
    std::optional<std::vector<int32_t>> queued_greedy_chain(const MtpTarget&,Tensor hidden,Tensor ids,
        Tensor positions,int depth,const std::function<void(const std::vector<int32_t>&)>& publish = {});

    int64_t cache_position() const noexcept override;

    void trim_cache_to(int64_t position) override;

    bool teacher_forced_prompt_prime() const noexcept override;

    bool target_bootstrap_decode() const noexcept override;

    bool preserve_output_dtype() const noexcept override;
    bool supports_session_state() const noexcept override;
    MtpSessionState capture_session_state(int64_t,const Tensor&) const override;
    void restore_session_state(const MtpSessionState&) override;
};

} // namespace mfq::cuda::qwen4_exp
