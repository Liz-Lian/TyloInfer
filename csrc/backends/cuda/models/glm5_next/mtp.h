#pragma once

#include "engine/executor/mtp.h"
#include "models/common/attention_ops.h"
#include "engine/models/glm5_next/config.h"

namespace mfq::cuda::weight_loader { struct Loader; }

namespace mfq::cuda::glm5_next {
using attention_ops::Linear;
class SparseMla;

struct Glm5NextMtp final : MtpModule {
    using Tensor = mfq_tensor_backend::Tensor;
    struct Layer {
        Tensor attention_norm, ffn_norm;
        std::unique_ptr<SparseMla> attention;
        attention_ops::Transform ffn;
    };

    mfq::models::glm5_next::Config config;
    Tensor embedding_norm, hidden_norm, output_norm;
    Linear fusion;
    std::vector<Layer> layers;
    std::vector<int64_t> lengths;
    int64_t batch = 0;

    Glm5NextMtp(Projector project, Collectives& copies, CudaProfiler& profiler,
                QuantLinear embedding, int primary);
    ~Glm5NextMtp() override;
    Glm5NextMtp(Glm5NextMtp&&) noexcept;

    static std::optional<Glm5NextMtp> load_if_present(const weight_loader::Loader& weights,
                                                      Projector projector, Collectives& copies,
                                                      CudaProfiler& profiler, QuantLinear embedding,
                                                      const mfq::models::glm5_next::Config &main);

    void reset(int64_t next_batch = 1) override;

    std::pair<Tensor, Tensor> evaluate(const Tensor &hidden,
                                       const Tensor &ids, int64_t depth = 0, bool cache = true,
                                       const Tensor &supplied_positions = {},
                                       const Tensor &supplied_embeddings = {});

    Tensor forward(Tensor hidden, Tensor ids) override;

    MtpStep step(Tensor hidden, Tensor ids) override;

    int64_t cache_position() const noexcept override;

    void trim_cache_to(int64_t position) override;

    bool teacher_forced_prompt_prime() const noexcept override;

    bool target_bootstrap_decode() const noexcept override;

    bool preserve_output_dtype() const noexcept override;

  private:
    Projector project_;
    Collectives& copies_;
    CudaProfiler& profiler_;
    QuantLinear embedding_;
    int primary_;
};

} // namespace mfq::cuda::glm5_next
