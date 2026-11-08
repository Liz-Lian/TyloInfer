#pragma once
#include "backend.h"
#include "engine/backends/cuda/ops/ffn/moe.h"
#include "engine/backends/cuda/executor/decode_window.h"
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <vector>
struct CudaProfiler;
struct KlMmqState;
struct MoeOptions;
namespace tyloi { struct MoeDispatchPlan; }
namespace tyloi::cuda { class Projector; class RouteStats; class Ffn; }

struct MoeFfnOptions {
    bool fused_activation = true;
    bool two_stage = false;
    bool resident_dispatch = false;
    bool reuse_input = true;
    std::optional<bool> transfer_phases;
};

using MoeFfnShared=std::function<::tyloi::DenseTensor(const tyloi::cuda::Projector&,const ::tyloi::DenseTensor&)>;
using MoeFfnDispatchObserver=std::function<void(const tyloi::MoeDispatchPlan&)>;
using MoeFfnPrefetch=std::function<tyloi::cuda::DecodeWindow::Task(const ::tyloi::DenseTensor&)>;
class MoeFfnPipeline;
struct MoeFfnSharedWeights {
    std::array<std::shared_ptr<const ::tyloi::Tensor>,3> projections;
    MoeFfnShared gate;
    std::shared_ptr<const ::tyloi::Tensor> scalar_gate;
};
namespace tyloi::cuda {

// Pipelines own mutable workspaces; models only borrow the resulting handles.
class Pipelines {
public:
    Pipelines(CudaProfiler& profiler, KlMmqState& mmq, const MoeOptions& kernels,
              const MoeFfnOptions& options, const RouteStats& routes)
        : profiler_(profiler), mmq_(mmq), kernels_(kernels), options_(options), routes_(routes) {}
    Pipelines(const Pipelines&) = delete;
    Pipelines& operator=(const Pipelines&) = delete;

    std::shared_ptr<MoeFfnPipeline> make(
        const std::vector<std::shared_ptr<const ::tyloi::Tensor>>& gate_up,
        const std::vector<std::shared_ptr<const ::tyloi::Tensor>>& down,
        MoeFfnShared shared = {}, MoeFfnDispatchObserver observer = {},
        MoeFfnSharedWeights shared_weights = {}, double swiglu_limit = 0) const;
    void add(std::shared_ptr<MoeFfnPipeline> pipeline);
    void prepare(Ffn& weights);
    void drain();
    void clear() noexcept { items_.clear(); }
    bool empty() const noexcept { return items_.empty(); }

private:
    CudaProfiler& profiler_;
    KlMmqState& mmq_;
    const MoeOptions& kernels_;
    const MoeFfnOptions& options_;
    const RouteStats& routes_;
    std::vector<std::shared_ptr<MoeFfnPipeline>> items_;
};

} // namespace tyloi::cuda

::tyloi::DenseTensor run_moe_ffn_pipeline(
    std::shared_ptr<MoeFfnPipeline> pipeline,const tyloi::cuda::Projector& project,
    const ::tyloi::DenseTensor& input,const ::tyloi::DenseTensor& ids,
    const ::tyloi::DenseTensor& weights,bool batch_serial=false,int tile=0,int tiles=1);
tyloi::cuda::DecodeWindow::Task prefetch_moe_ffn_pipeline(
    std::shared_ptr<MoeFfnPipeline> pipeline,const ::tyloi::DenseTensor& ids);
bool moe_ffn_neighbor_prefetch_enabled(const std::shared_ptr<MoeFfnPipeline>& pipeline);
#ifdef TYLOINFER_NATIVE_CUDA
bool moe_ffn_use_pipeline(const std::shared_ptr<MoeFfnPipeline>& pipeline,
                          std::int64_t rows, std::size_t gate_up_count);
::tyloi::DenseTensor run_moe_ffn_pipeline(const std::shared_ptr<MoeFfnPipeline>& pipeline,
    const ::tyloi::DenseTensor& input, const ::tyloi::DenseTensor& ids,
    const ::tyloi::DenseTensor& weights);
#endif
void activate_moe_ffn_pipeline(std::shared_ptr<MoeFfnPipeline> pipeline);
void drain_moe_ffn_pipeline(std::shared_ptr<MoeFfnPipeline> pipeline);
