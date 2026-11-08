#pragma once

#include "engine/models/common/mtp.h"
#include "engine/models/session.h"
#include "engine/models/common/attention_ops.h"
#include "engine/models/qwen4_exp/config.h"

struct CudaExecutionContext;
namespace tyloi::weights { struct Loader; }

namespace tyloi::cuda { class GatedResidual; }

namespace tyloi::cuda::qwen4_exp {
using attention_ops::Linear;
struct Qwen4Block;
struct LayerState;
struct Qwen4MtpGraph;

struct Predictor final : tyloi::models::Mtp {
    using DenseTensor = ::tyloi::DenseTensor;

    tyloi::models::qwen4_exp::Config config;
    std::shared_ptr<const ::tyloi::Tensor> embedding, output;
    double softcap = 0;
    int primary = 0;
#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<const ::tyloi::Tensor> embedding_norm, hidden_norm;
#else
    DenseTensor embedding_norm, hidden_norm;
#endif
    Linear embedding_fusion, hidden_fusion;
    std::unique_ptr<GatedResidual> final_mixer;
    std::vector<std::unique_ptr<Qwen4Block>> layers;
    std::vector<std::unique_ptr<LayerState>> layer_states;
    std::vector<DenseTensor> positions;
    std::vector<int64_t> lengths;
    CudaExecutionContext *execution = nullptr;
    int64_t batch = 0;
    std::unique_ptr<Qwen4MtpGraph> graph_registry;
    bool reuse_decode_graphs=false;
    uint64_t graph_captures=0;
    uint64_t graph_replays = 0;
    bool graph_initialized = false;
    bool compact_graph_sampling = true;
    DenseTensor graph_compact_topk() const;
    bool take_graph_initialization() noexcept override { return std::exchange(graph_initialized,false); }
    uint64_t graph_replay_count() const noexcept override { return graph_replays; }

    Predictor();
    ~Predictor() override;
    Predictor(Predictor&&) noexcept;
    Predictor& operator=(Predictor&&) noexcept;

    static std::optional<Predictor> load_if_present(CudaExecutionContext &execution,
                                                      const tyloi::weights::Loader& weights,
                                                      const tyloi::models::qwen4_exp::Config &main,
                                                      std::shared_ptr<const ::tyloi::Tensor> embedding, std::shared_ptr<const ::tyloi::Tensor> output,
                                                      double softcap);

    DenseTensor embed(DenseTensor ids) const;
    DenseTensor logits(DenseTensor hidden) const;

    void reset(int64_t next_batch = 1) override;

    std::pair<DenseTensor, DenseTensor> evaluate(const DenseTensor &hidden,
                                       const DenseTensor &ids, int64_t depth = 0, bool cache = true,
                                       const DenseTensor &supplied_positions = {},
                                       const DenseTensor &supplied_embeddings = {},
                                       bool history_only = false);

    tyloi::models::MtpStep step(const tyloi::Tensor& hidden, const tyloi::Tensor& ids,
                                const tyloi::Tensor* positions = nullptr) override;
    void history_step(DenseTensor hidden, DenseTensor ids, DenseTensor positions = {},
                      bool scalar_chain = false);

    int64_t cache_position() const noexcept override;

    void trim_cache_to(int64_t position) override;

    bool teacher_forced_prompt_prime() const noexcept override;

    bool target_bootstrap_decode() const noexcept override;

    bool preserve_output_dtype() const noexcept override;
    std::unique_ptr<tyloi::engine::mtp::DepthPolicy> depth_policy(
        int maximum_depth, tyloi::engine::mtp::PolicyState& state) const override;
    bool supports_session_state() const noexcept override;
    MtpSessionState capture_session_state(int64_t,const tyloi::Tensor&) const override;
    void restore_session_state(const MtpSessionState&) override;
};

} // namespace tyloi::cuda::qwen4_exp
