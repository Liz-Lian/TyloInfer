#pragma once
#include "engine/weights/moe/experts.h"
#include "engine/weights/dispatch_plan.h"
#include "backends/cuda/engine/executor/decode_window.h"
#include <array>
#include <memory>
#include <vector>
struct CudaExecutionContext;
using MoeFfnShared=std::function<mfq_tensor_backend::Tensor(CudaExecutionContext&,const mfq_tensor_backend::Tensor&)>;
using MoeFfnDispatchObserver=std::function<void(const mfq::MoeDispatchPlan&)>;
using MoeFfnPrefetch=std::function<mfq::cuda::DecodeWindow::Task(const mfq_tensor_backend::Tensor&)>;
class MoeFfnPipeline;
struct QuantLinear;
struct MoeFfnSharedWeights {
    std::array<std::shared_ptr<const QuantLinear>,3> projections;
    MoeFfnShared gate;
    std::shared_ptr<const QuantLinear> scalar_gate;
};
std::shared_ptr<MoeFfnPipeline> make_moe_ffn_pipeline(const std::vector<std::shared_ptr<mfq::cuda::Experts>>& gate_up,
    const std::vector<std::shared_ptr<mfq::cuda::Experts>>& down,MoeFfnShared shared={},MoeFfnDispatchObserver observer={},
    MoeFfnSharedWeights shared_weights={},double swiglu_limit=0);
mfq_tensor_backend::Tensor run_moe_ffn_pipeline(
    std::shared_ptr<MoeFfnPipeline> pipeline,CudaExecutionContext& execution,
    const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& ids,
    const mfq_tensor_backend::Tensor& weights);
mfq::cuda::DecodeWindow::Task prefetch_moe_ffn_pipeline(
    std::shared_ptr<MoeFfnPipeline> pipeline,const mfq_tensor_backend::Tensor& ids);
void drain_moe_ffn_pipeline(std::shared_ptr<MoeFfnPipeline> pipeline);
