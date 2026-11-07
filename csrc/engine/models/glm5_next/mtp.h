#pragma once

#include "engine/models/common/mtp.h"
#include "engine/models/common/attention_ops.h"
#include "engine/models/glm5_next/config.h"
#include "engine/models/glm5_next/state.h"

namespace tyloi::weights { struct Loader; }

namespace tyloi::cuda::glm5_next {
using attention_ops::Linear;
class SparseMla;

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
    std::vector<MlaState> states;
    std::vector<int64_t> lengths;
    int64_t batch = 0;

    Mtp(Projector project, Collectives& copies, CudaProfiler& profiler,
                std::shared_ptr<const ::tyloi::Tensor> embedding, int primary);
    ~Mtp() override;
    Mtp(Mtp&&) noexcept;

    static std::optional<Mtp> load_if_present(const ::tyloi::weights::Loader& weights,
                                                      Projector projector, Collectives& copies,
                                                      CudaProfiler& profiler, std::shared_ptr<const ::tyloi::Tensor> embedding,
                                                      const tyloi::models::glm5_next::Config &main);

    void reset(int64_t next_batch = 1) override;

    std::pair<DenseTensor, DenseTensor> evaluate(const DenseTensor &hidden,
                                       const DenseTensor &ids, int64_t depth = 0, bool cache = true,
                                       const DenseTensor &supplied_positions = {},
                                       const DenseTensor &supplied_embeddings = {});

    tyloi::models::MtpStep step(const ::tyloi::Tensor& hidden, const ::tyloi::Tensor& ids,
                                const ::tyloi::Tensor* positions = nullptr) override;

    int64_t cache_position() const noexcept override;

    void trim_cache_to(int64_t position) override;

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
