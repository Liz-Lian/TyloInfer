#pragma once

#include "engine/models/qwen4_exp/config.h"
#include "engine/backends/cuda/executor/moe_ffn_pipeline.h"

#include <array>
#include <memory>
#include <vector>

struct CudaExecutionContext;
struct TqMoeBanks;

namespace tyloi::cuda::qwen4_exp {

class Moe {
  public:
    Moe() = default;
    Moe(Pipelines&, std::vector<::tyloi::Tensor> gate_up, ::tyloi::Tensor down,
        ::tyloi::Tensor router, ::tyloi::Tensor shared_gate, ::tyloi::Tensor gate,
        ::tyloi::Tensor up, ::tyloi::Tensor shared_down,
        const tyloi::models::qwen4_exp::Config&, MoeFfnPrefetch* = nullptr);
    ::tyloi::DenseTensor forward(CudaExecutionContext&, const ::tyloi::DenseTensor&) const;

  private:
    struct Shared {
        std::array<::tyloi::Tensor, 3> projections;
        ::tyloi::Tensor scalar_gate;
        bool tq = false, row_parallel = false;
        ::tyloi::DenseTensor raw(const Projector&, const ::tyloi::DenseTensor&) const;
        ::tyloi::DenseTensor forward(const Projector&, const ::tyloi::DenseTensor&) const;
        ::tyloi::DenseTensor gate(const Projector&, const ::tyloi::DenseTensor&) const;
    };
    std::vector<::tyloi::Tensor> gate_up_;
    ::tyloi::Tensor down_, router_;
    tyloi::models::qwen4_exp::Config config_{};
    std::shared_ptr<const Shared> shared_;
    std::shared_ptr<TqMoeBanks> tq_;
    std::weak_ptr<MoeFfnPipeline> pipeline_;
};

} // namespace tyloi::cuda::qwen4_exp
