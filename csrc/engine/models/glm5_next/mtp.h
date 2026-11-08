#pragma once

#include "engine/models/common/mtp.h"
#include "engine/models/common/attention_ops.h"
#include "engine/models/glm5_next/config.h"
#include "engine/models/glm5_next/state.h"

namespace tyloi::weights { struct Loader; }

namespace tyloi::cuda::glm5_next {
using attention_ops::Linear;
class SparseMla;

struct PredictorState final : tyloi::models::MtpState {
    explicit PredictorState(const tyloi::models::Mtp& owner) : MtpState(owner) {}

    int64_t position() const noexcept override {
        return lengths.empty() ? 0 : lengths.front();
    }

    std::vector<MlaState> states;
    std::vector<int64_t> lengths;
    int64_t batch = 0;
};

struct Mtp final : tyloi::models::Mtp {
    using DenseTensor = ::tyloi::DenseTensor;
#ifdef TYLOINFER_NATIVE_CUDA
    using Parameter = std::shared_ptr<const ::tyloi::Tensor>;
#else
    using Parameter = DenseTensor;
#endif
    struct Layer {
        Parameter attention_norm, ffn_norm;
        std::unique_ptr<SparseMla> attention;
        attention_ops::Transform ffn;
    };

    tyloi::models::glm5_next::Config config;
    Parameter embedding_norm, hidden_norm, output_norm;
    Linear fusion;
    std::vector<Layer> layers;

    Mtp(Projector project, Collectives& copies, CudaProfiler& profiler,
                std::shared_ptr<const ::tyloi::Tensor> embedding, int primary);
    ~Mtp() override;
    Mtp(Mtp&&) noexcept;

    static std::optional<Mtp> load_if_present(const ::tyloi::weights::Loader& weights,
                                                      Projector projector, Collectives& copies,
                                                      CudaProfiler& profiler, std::shared_ptr<const ::tyloi::Tensor> embedding,
                                                      const tyloi::models::glm5_next::Config &main);

    std::unique_ptr<tyloi::models::MtpState> make_state() const override {
        auto result = std::make_unique<PredictorState>(*this);
        result->states.reserve(layers.size());
        for (size_t i = 0; i < layers.size(); ++i)
            result->states.emplace_back(config.maximum, config.latent, config.index_width);
        result->lengths.assign(layers.size(), 0);
        return result;
    }

    void reset(tyloi::models::MtpState& state, int64_t next_batch = 1) const override;

    std::pair<DenseTensor, DenseTensor> evaluate(tyloi::models::MtpState& state,
                                       const DenseTensor &hidden,
                                       const DenseTensor &ids, int64_t depth = 0, bool cache = true,
                                       const DenseTensor &supplied_positions = {},
                                       const DenseTensor &supplied_embeddings = {}) const;

    tyloi::models::MtpStep step(tyloi::models::MtpState& state,
                                const ::tyloi::Tensor& hidden, const ::tyloi::Tensor& ids,
                                const ::tyloi::Tensor& positions = {}) const override;

    void trim_cache_to(tyloi::models::MtpState& state, int64_t position) const override;

    bool teacher_forced_prompt_prime() const noexcept override;

    bool target_bootstrap_decode() const noexcept override;

    bool preserve_output_dtype() const noexcept override;

  private:
    Projector project_;
    Collectives& copies_;
    CudaProfiler& profiler_;
    std::shared_ptr<const ::tyloi::Tensor> embedding_;
    int primary_;
};

} // namespace tyloi::cuda::glm5_next
