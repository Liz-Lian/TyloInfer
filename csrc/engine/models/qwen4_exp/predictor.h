#pragma once

#include "engine/models/common/mtp.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/common/attention_ops.h"
#include "engine/models/qwen4_exp/config.h"

namespace tyloi::weights { struct Loader; }

namespace tyloi::ops { class Residual; }

namespace tyloi::cuda::qwen4_exp {
using attention_ops::Linear;
struct Layer;
struct LayerState;
struct Predictor;

struct PredictorState final : tyloi::models::MtpState {
    explicit PredictorState(const Predictor& owner);
    ~PredictorState() override;
    int64_t position() const noexcept override;
    /** Return the committed boundary of each predictor head. */
    std::vector<std::int64_t> head_positions() const override { return lengths; }

    std::vector<std::unique_ptr<LayerState>> layer_states;
    std::vector<::tyloi::Tensor> positions;
    std::vector<int64_t> lengths;
    int64_t batch = 0;
};

struct Predictor final : tyloi::models::Mtp {
    using DenseTensor = ::tyloi::DenseTensor;

    tyloi::models::qwen4_exp::Config config;
    ::tyloi::Tensor embedding, output;
    double softcap = 0;
    int primary = 0;
    ::tyloi::Tensor embedding_norm, hidden_norm;
    Linear embedding_fusion, hidden_fusion;
    std::unique_ptr<tyloi::ops::Residual> final_mixer;
    std::vector<std::unique_ptr<Layer>> layers;
    bool history_stream = true, history_bootstrap = true, history_early_flush = true, history_cache_only = true;
    int history_host = -1;

    Predictor();
    ~Predictor() override;
    Predictor(Predictor&&) noexcept;
    Predictor& operator=(Predictor&&) noexcept;

    /** Load predictor weights, if present, using the model's rotary policy. */
    static std::optional<Predictor> load_if_present(const tyloi::weights::Loader& weights,
                                                      const tyloi::models::qwen4_exp::Config &main,
                                                      ::tyloi::Tensor embedding, ::tyloi::Tensor output,
                                                      double softcap, const tyloi::ops::YarnScaling& yarn);

    DenseTensor embed(DenseTensor ids) const;
    DenseTensor logits(DenseTensor hidden) const;

    std::unique_ptr<tyloi::models::MtpState> make_state() const override;
    /** Return the incoming multistream predictor width. */
    std::int64_t width() const override { return hidden_norm.numel(); }
    void reset(tyloi::models::MtpState&, int64_t next_batch = 1) const override;
    /** Plan persistent cache and position backing for a predictor step. */
    std::vector<std::pair<::tyloi::Device, std::size_t>> growth(
        const tyloi::models::MtpState&, const tyloi::models::Mtp::Input&) const override;

    std::pair<DenseTensor, DenseTensor> evaluate(tyloi::models::MtpState&, const DenseTensor &hidden,
                                       const DenseTensor &ids, int64_t depth = 0, bool cache = true,
                                       const DenseTensor &supplied_positions = {},
                                       const DenseTensor &supplied_embeddings = {},
                                       bool history_only = false) const;
    /** Run the selected head through the original eager predictor path. */
    tyloi::models::MtpStep evaluate(tyloi::models::MtpState&,
        const tyloi::models::Mtp::Evaluation&) const override;

    tyloi::models::MtpStep step(tyloi::models::MtpState&, const tyloi::Tensor& hidden, const tyloi::Tensor& ids,
                                const tyloi::Tensor& positions = {},
                                const tyloi::models::ForwardOptions& options = {}) const override;
    /** Resolve prompt history behavior from frozen model policy and available memory. */
    History history(std::int64_t tokens, std::int64_t requested, std::size_t available) const override;
    /** Prime predictor history while optionally omitting output computation. */
    void prime(tyloi::models::MtpState&, const tyloi::Tensor& hidden, const tyloi::Tensor& ids,
        const tyloi::Tensor& positions, bool cache_only,
        const tyloi::models::ForwardOptions& options = {}) const override;
    void history_step(tyloi::models::MtpState&, DenseTensor hidden, DenseTensor ids, DenseTensor positions = {},
                      bool scalar_chain = false) const;

    void trim_cache_to(tyloi::models::MtpState&, int64_t position) const override;

    bool teacher_forced_prompt_prime() const noexcept override;

    bool target_bootstrap_decode() const noexcept override;

    bool preserve_output_dtype() const noexcept override;
    std::unique_ptr<tyloi::engine::mtp::DepthPolicy> depth_policy(
        int maximum_depth, tyloi::engine::mtp::PolicyState& state) const override;
    /** Check session support against the owning predictor request state. */
    bool supports_session_state(const tyloi::models::MtpState&) const noexcept override;
    MtpSessionState capture_session_state(const tyloi::models::MtpState&, int64_t,const tyloi::Tensor&) const override;
    /** Prepare an owned predictor snapshot for a joint session restore. */
    std::unique_ptr<tyloi::models::Restore> prepare_restore(
        tyloi::models::MtpState&, const MtpSessionState&) const override;
};

} // namespace tyloi::cuda::qwen4_exp
