#pragma once

#include "engine/models/common/mtp.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/common/attention_ops.h"
#include "engine/models/qwen4_exp/config.h"

struct CudaExecutionContext;
namespace tyloi::weights { struct Loader; }

namespace tyloi::cuda { class GatedResidual; }

namespace tyloi::cuda::qwen4_exp {
using attention_ops::Linear;
struct Layer;
struct LayerState;
struct Predictor;

struct PredictorState final : tyloi::models::MtpState {
    explicit PredictorState(const Predictor& owner);
    ~PredictorState() override;
    int64_t position() const noexcept override;

    std::vector<std::unique_ptr<LayerState>> layer_states;
    std::vector<::tyloi::DenseTensor> positions;
    std::vector<int64_t> lengths;
    int64_t batch = 0;
};

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
    std::vector<std::unique_ptr<Layer>> layers;
    CudaExecutionContext *execution = nullptr;

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

    std::unique_ptr<tyloi::models::MtpState> make_state() const override;
    void reset(tyloi::models::MtpState&, int64_t next_batch = 1) const override;

    std::pair<DenseTensor, DenseTensor> evaluate(tyloi::models::MtpState&, const DenseTensor &hidden,
                                       const DenseTensor &ids, int64_t depth = 0, bool cache = true,
                                       const DenseTensor &supplied_positions = {},
                                       const DenseTensor &supplied_embeddings = {},
                                       bool history_only = false) const;

    tyloi::models::MtpStep step(tyloi::models::MtpState&, const tyloi::Tensor& hidden, const tyloi::Tensor& ids,
                                const tyloi::Tensor& positions = {}) const override;
    void history_step(tyloi::models::MtpState&, DenseTensor hidden, DenseTensor ids, DenseTensor positions = {},
                      bool scalar_chain = false) const;

    void trim_cache_to(tyloi::models::MtpState&, int64_t position) const override;

    bool teacher_forced_prompt_prime() const noexcept override;

    bool target_bootstrap_decode() const noexcept override;

    bool preserve_output_dtype() const noexcept override;
    std::unique_ptr<tyloi::engine::mtp::DepthPolicy> depth_policy(
        int maximum_depth, tyloi::engine::mtp::PolicyState& state) const override;
    bool supports_session_state() const noexcept override;
    MtpSessionState capture_session_state(const tyloi::models::MtpState&, int64_t,const tyloi::Tensor&) const override;
    void restore_session_state(tyloi::models::MtpState&, const MtpSessionState&) const override;
};

} // namespace tyloi::cuda::qwen4_exp
