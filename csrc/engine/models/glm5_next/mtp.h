#pragma once

#include "engine/models/common/mtp.h"
#include "engine/models/common/attention_ops.h"
#include "engine/models/glm5_next/config.h"
#include "engine/models/glm5_next/state.h"

namespace tyloi::cuda::weight_loader { struct Loader; }

namespace tyloi::cuda::glm5_next {
using attention_ops::Linear;
class SparseMla;

struct Mtp final : MtpModule {
    using DenseTensor = ::tyloi::DenseTensor;
    struct Layer {
        DenseTensor attention_norm, ffn_norm;
        std::unique_ptr<SparseMla> attention;
        attention_ops::Transform ffn;
    };

    tyloi::models::glm5_next::Config config;
    DenseTensor embedding_norm, hidden_norm, output_norm;
    Linear fusion;
    std::vector<Layer> layers;
    std::vector<MlaState> states;
    std::vector<int64_t> lengths;
    int64_t batch = 0;

    Mtp(Projector project, Collectives& copies, CudaProfiler& profiler,
                QuantLinear embedding, int primary);
    ~Mtp() override;
    Mtp(Mtp&&) noexcept;

    static std::optional<Mtp> load_if_present(const weight_loader::Loader& weights,
                                                      Projector projector, Collectives& copies,
                                                      CudaProfiler& profiler, QuantLinear embedding,
                                                      const tyloi::models::glm5_next::Config &main);

    void reset(int64_t next_batch = 1) override;

    std::pair<DenseTensor, DenseTensor> evaluate(const DenseTensor &hidden,
                                       const DenseTensor &ids, int64_t depth = 0, bool cache = true,
                                       const DenseTensor &supplied_positions = {},
                                       const DenseTensor &supplied_embeddings = {});

    DenseTensor forward(DenseTensor hidden, DenseTensor ids) override;

    MtpStep step(DenseTensor hidden, DenseTensor ids) override;

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

} // namespace tyloi::cuda::glm5_next
