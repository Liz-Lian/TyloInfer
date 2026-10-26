#pragma once
#include "backend.h"
#include "backends/cuda/engine/executor/decode_window.h"
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <vector>
struct CudaProfiler;
struct KlMmqState;
struct MoeOptions;
namespace mfq { struct MoeDispatchPlan; }
namespace mfq::cuda { class Projector; struct Experts; }

struct MoeFfnOptions {
    bool fused_activation = true;
    bool two_stage = false;
    bool resident_dispatch = false;
    bool reuse_input = true;
    std::optional<bool> transfer_phases;
};

using MoeFfnShared=std::function<mfq_tensor_backend::Tensor(const mfq::cuda::Projector&,const mfq_tensor_backend::Tensor&)>;
using MoeFfnDispatchObserver=std::function<void(const mfq::MoeDispatchPlan&)>;
using MoeFfnPrefetch=std::function<mfq::cuda::DecodeWindow::Task(const mfq_tensor_backend::Tensor&)>;
class MoeFfnPipeline;
struct QuantLinear;
struct MoeFfnSharedWeights {
    std::array<std::shared_ptr<const QuantLinear>,3> projections;
    MoeFfnShared gate;
    std::shared_ptr<const QuantLinear> scalar_gate;
};
// Borrowed resources and live options outlive the pipeline and its drain.
std::shared_ptr<MoeFfnPipeline> make_moe_ffn_pipeline(
    CudaProfiler& profiler, KlMmqState& mmq, const MoeOptions& kernels, const MoeFfnOptions& options,
    const std::vector<std::shared_ptr<mfq::cuda::Experts>>& gate_up,
    const std::vector<std::shared_ptr<mfq::cuda::Experts>>& down,MoeFfnShared shared={},MoeFfnDispatchObserver observer={},
    MoeFfnSharedWeights shared_weights={},double swiglu_limit=0);
mfq_tensor_backend::Tensor run_moe_ffn_pipeline(
    std::shared_ptr<MoeFfnPipeline> pipeline,const mfq::cuda::Projector& project,
    const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& ids,
    const mfq_tensor_backend::Tensor& weights,bool batch_serial=false,int tile=0,int tiles=1);
mfq::cuda::DecodeWindow::Task prefetch_moe_ffn_pipeline(
    std::shared_ptr<MoeFfnPipeline> pipeline,const mfq_tensor_backend::Tensor& ids);
void activate_moe_ffn_pipeline(std::shared_ptr<MoeFfnPipeline> pipeline);
void drain_moe_ffn_pipeline(std::shared_ptr<MoeFfnPipeline> pipeline);
