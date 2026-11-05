#pragma once

#include "quant_linear.h"
#include "engine/backends/cuda/ops/ffn/moe.h"

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
namespace tyloi::cuda::ops {
tyloi::DenseTensor embedding(tyloi::DenseTensor ids, const ::tyloi::Tensor& weight);
tyloi::DenseTensor project(tyloi::DenseTensor input, const ::tyloi::Tensor& weight);
tyloi::DenseTensor linear(tyloi::DenseTensor input, const ::tyloi::Tensor& weight, const tyloi::DenseTensor* bias);
tyloi::DenseTensor affine(tyloi::DenseTensor input, const ::tyloi::Tensor& weight, const tyloi::DenseTensor* bias);
std::vector<tyloi::DenseTensor> project_group(tyloi::DenseTensor input, const ::QuantLinearGroup& group);
tyloi::DenseTensor sigmoid_project(tyloi::DenseTensor input, tyloi::DenseTensor gate, const ::tyloi::Tensor& weight);
}
#endif

struct TqMoeBanks;

namespace tyloi::cuda {

class Branches;
struct Experts;

struct Projections {
    std::vector<std::shared_ptr<const ::tyloi::Tensor>> weights;
    QuantLinearGroup group;
    std::vector<size_t> indices;
    size_t prefix = 1;
    bool joined = false;

    bool grouped() const noexcept {
        return group.tq.defined() || group.nint_grouped;
    }
};

// Bound resources and options must outlive this object; drain before resetting them.
class Projector {
public:
    struct Options {
        bool peer_first = true;
        bool reduce_to_primary = true;
        bool fp16_reduce = true;
        bool grouped = true;
        bool parallel = true;
        bool nvq_fusion = true;
        MoeOptions moe;
    };

    Projector(CudaProfiler& profiler, KlMmqState& mmq, TqExec& tq,
              Branches& branches, Collectives& collectives, MfeScratch& mfe_scratch,
              MoeScratchCache& moe_scratch, const Options& options,
              int primary, const bool& serial, const bool& projection_major)
        : profiler_(profiler), mmq_(mmq), tq_(tq), branches_(branches),
          collectives_(collectives), mfe_scratch_(mfe_scratch), moe_scratch_(moe_scratch), options_(options), primary_(primary),
          serial_(serial), projection_major_(projection_major) {}

    int primary_device() const { return primary_; }
    MoeScratchCache& moe_scratch() const { return moe_scratch_; }

    ::tyloi::DenseTensor run(const ::tyloi::Tensor& weight, ::tyloi::DenseTensor input) const;
    std::vector<::tyloi::DenseTensor> run(
        const Projections& projections, ::tyloi::DenseTensor input) const;
    std::vector<::tyloi::DenseTensor> run(
        const QuantLinearGroup& group, ::tyloi::DenseTensor input) const;
    ::tyloi::DenseTensor gated(const ::tyloi::Tensor& weight, ::tyloi::DenseTensor input,
                                    ::tyloi::DenseTensor gate, InputGateMode mode) const;
    ::tyloi::DenseTensor groupwise(const ::tyloi::Tensor& weight,
                                       ::tyloi::DenseTensor input, int64_t groups) const;
    ::tyloi::DenseTensor low_rank(::tyloi::DenseTensor attention,
                                      const ::tyloi::Tensor& output_a, const ::tyloi::Tensor& output_b,
                                      int64_t groups, bool groupwise, bool profile) const;
    ::tyloi::DenseTensor swiglu(const QuantLinearGroup& group, ::tyloi::DenseTensor input) const;
    ::tyloi::DenseTensor geglu(const QuantLinearGroup& group, ::tyloi::DenseTensor input) const;
    std::vector<::tyloi::DenseTensor> grouped(
        const std::vector<const ::tyloi::Tensor*>& projections, ::tyloi::DenseTensor input) const;

    ::tyloi::DenseTensor run(
        const Experts& weight, ::tyloi::DenseTensor input, const MoeRoutePlan& route,
        bool prequantized = false, int epilogue = 0) const;
    ::tyloi::DenseTensor glu(
        const Experts& weight, ::tyloi::DenseTensor input, const MoeRoutePlan& route,
        bool gelu) const;
    ::tyloi::DenseTensor gated(
        const Experts& weight, ::tyloi::DenseTensor input, const MoeRoutePlan& route,
        bool gelu) const;
    ::tyloi::DenseTensor clamped(
        const Experts& weight, ::tyloi::DenseTensor input, const MoeRoutePlan& route,
        double limit) const;
    ::tyloi::DenseTensor run(
        const TqMoeBanks& banks, ::tyloi::DenseTensor input,
        ::tyloi::DenseTensor ids, ::tyloi::DenseTensor weights,
        float swiglu_limit = 0) const;

private:
    CudaProfiler& profiler_;
    KlMmqState& mmq_;
    TqExec& tq_;
    Branches& branches_;
    Collectives& collectives_;
    MfeScratch& mfe_scratch_;
    MoeScratchCache& moe_scratch_;
    const Options& options_;
    int primary_;
    const bool& serial_;
    const bool& projection_major_;

    ::tyloi::DenseTensor sharded(const ::tyloi::ShardedTensor& weight, ::tyloi::DenseTensor input,
                                      MfqOptional<::tyloi::DenseTensor> gate, InputGateMode mode) const;
    std::vector<::tyloi::DenseTensor> run(
        const NintLinearGroup& group, ::tyloi::DenseTensor input) const;
    ::tyloi::DenseTensor swiglu(const NintLinearGroup& group, ::tyloi::DenseTensor input) const;
    ::tyloi::DenseTensor geglu(const NintLinearGroup& group, ::tyloi::DenseTensor input) const;
};

::tyloi::DenseTensor embed(const ::tyloi::Tensor& weight, ::tyloi::DenseTensor ids,
    Collectives& copies, CudaProfiler& profiler, int device);
::tyloi::DenseTensor logits(const ::tyloi::Tensor& weight, ::tyloi::DenseTensor hidden,
    const Projector& project, CudaProfiler& profiler, double softcap);

} // namespace tyloi::cuda
