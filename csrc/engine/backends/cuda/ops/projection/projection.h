#pragma once

#include "quant_linear.h"
#include "engine/backends/cuda/ops/ffn/moe.h"

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
namespace tyloi::cuda::ops {
tyloi::Tensor embedding(tyloi::Tensor ids, const weights::Weights& weight);
tyloi::Tensor project(tyloi::Tensor input, const weights::Weights& weight);
tyloi::Tensor linear(tyloi::Tensor input, const weights::Weights& weight, const tyloi::Tensor& bias);
tyloi::Tensor affine(tyloi::Tensor input, const weights::Weights& weight, const tyloi::Tensor& bias);
std::vector<tyloi::Tensor> project_group(tyloi::Tensor input, const weights::Weights& group);
tyloi::Tensor sigmoid_project(tyloi::Tensor input, tyloi::Tensor gate, const weights::Weights& weight);
}
#endif

struct TqMoeBanks;

namespace tyloi::cuda {

class Branches;
struct Experts;

struct Projections {
    std::vector<QuantLinear> weights;
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

    ::tyloi::Tensor run(const QuantLinear& weight, ::tyloi::Tensor input) const;
    std::vector<::tyloi::Tensor> run(
        const Projections& projections, ::tyloi::Tensor input) const;
    std::vector<::tyloi::Tensor> run(
        const QuantLinearGroup& group, ::tyloi::Tensor input) const;
    ::tyloi::Tensor gated(const QuantLinear& weight, ::tyloi::Tensor input,
                                    ::tyloi::Tensor gate, InputGateMode mode) const;
    ::tyloi::Tensor groupwise(const QuantLinear& weight,
                                       ::tyloi::Tensor input, int64_t groups) const;
    ::tyloi::Tensor low_rank(::tyloi::Tensor attention,
                                      const QuantLinear& output_a, const QuantLinear& output_b,
                                      int64_t groups, bool groupwise, bool profile) const;
    ::tyloi::Tensor swiglu(const QuantLinearGroup& group, ::tyloi::Tensor input) const;
    ::tyloi::Tensor geglu(const QuantLinearGroup& group, ::tyloi::Tensor input) const;
    std::vector<::tyloi::Tensor> grouped(
        const QuantLinearProjectionRefs& projections, ::tyloi::Tensor input) const;

    ::tyloi::Tensor run(
        const Experts& weight, ::tyloi::Tensor input, const MoeRoutePlan& route,
        bool prequantized = false, int epilogue = 0) const;
    ::tyloi::Tensor glu(
        const Experts& weight, ::tyloi::Tensor input, const MoeRoutePlan& route,
        bool gelu) const;
    ::tyloi::Tensor gated(
        const Experts& weight, ::tyloi::Tensor input, const MoeRoutePlan& route,
        bool gelu) const;
    ::tyloi::Tensor clamped(
        const Experts& weight, ::tyloi::Tensor input, const MoeRoutePlan& route,
        double limit) const;
    ::tyloi::Tensor run(
        const TqMoeBanks& banks, ::tyloi::Tensor input,
        ::tyloi::Tensor ids, ::tyloi::Tensor weights,
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

    ::tyloi::Tensor sharded(const QuantLinear& weight, ::tyloi::Tensor input,
                                      MfqOptional<::tyloi::Tensor> gate, InputGateMode mode) const;
    std::vector<::tyloi::Tensor> run(
        const NintLinearGroup& group, ::tyloi::Tensor input) const;
    ::tyloi::Tensor swiglu(const NintLinearGroup& group, ::tyloi::Tensor input) const;
    ::tyloi::Tensor geglu(const NintLinearGroup& group, ::tyloi::Tensor input) const;
};

::tyloi::Tensor embed(const QuantLinear& weight, ::tyloi::Tensor ids,
    Collectives& copies, CudaProfiler& profiler, int device);
::tyloi::Tensor logits(const QuantLinear& weight, ::tyloi::Tensor hidden,
    const Projector& project, CudaProfiler& profiler, double softcap);

} // namespace tyloi::cuda
