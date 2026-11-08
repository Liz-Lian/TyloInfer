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
        const tyloi::models::qwen4_exp::Config&);
    const auto& router() const noexcept { return router_; }
    bool tq() const noexcept {
#ifdef TYLOINFER_NATIVE_CUDA
        return tq_;
#else
        return bool(tq_);
#endif
    }
    bool can_prefetch() const noexcept { return !tq() && !pipeline_.expired(); }
    ::tyloi::cuda::DecodeWindow::Task prefetch(const ::tyloi::DenseTensor& ids) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if (tq()) return {};
        if (auto pipeline = pipeline_.lock())
            return prefetch_moe_ffn_pipeline(pipeline, ids);
#endif
        return {};
    }
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::DenseTensor forward(const ::tyloi::DenseTensor&) const;
#else
    ::tyloi::DenseTensor forward(CudaExecutionContext&, const ::tyloi::DenseTensor&) const;
#endif

  private:
    std::vector<::tyloi::Tensor> gate_up_;
    ::tyloi::Tensor down_, router_;
    tyloi::models::qwen4_exp::Config config_{};
    std::shared_ptr<const MoeSharedExpert> shared_;
#ifdef TYLOINFER_NATIVE_CUDA
    bool tq_ = false;
#else
    std::shared_ptr<TqMoeBanks> tq_;
#endif
    std::weak_ptr<MoeFfnPipeline> pipeline_;
};

} // namespace tyloi::cuda::qwen4_exp
