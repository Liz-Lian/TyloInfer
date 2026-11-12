#pragma once

#include "quant_linear.h"
#include "engine/backends/cuda/ops/ffn/moe.h"

#ifdef TYLOINFER_NATIVE_CUDA
namespace tyloi::cuda::ops {
tyloi::DenseTensor embedding(tyloi::DenseTensor ids, const ::tyloi::Tensor& weight);
tyloi::DenseTensor project(tyloi::DenseTensor input, const ::tyloi::Tensor& weight);
tyloi::DenseTensor project(tyloi::DenseTensor input, const ::tyloi::Tensor& weight,
                           tyloi::DenseTensor ids);
tyloi::DenseTensor low_rank(tyloi::DenseTensor input, const ::tyloi::Tensor& output_a,
                           const ::tyloi::Tensor& output_b, int64_t groups, bool groupwise, bool profile);
tyloi::DenseTensor linear(tyloi::DenseTensor input, const ::tyloi::Tensor& weight, const tyloi::DenseTensor* bias);
tyloi::DenseTensor affine(tyloi::DenseTensor input, const ::tyloi::Tensor& weight, const tyloi::DenseTensor* bias);
std::vector<tyloi::DenseTensor> project_group(tyloi::DenseTensor input, std::span<const ::tyloi::Tensor> group);
bool grouped(std::span<const ::tyloi::Tensor> weights);
std::vector<tyloi::DenseTensor> project_group(tyloi::DenseTensor input,
    std::span<const ::tyloi::Tensor> weights, std::size_t prefix);
tyloi::DenseTensor sigmoid_project(tyloi::DenseTensor input, tyloi::DenseTensor gate, const ::tyloi::Tensor& weight);
}
#endif

struct TqMoeBanks;

namespace tyloi::cuda {

class Branches;

DenseLinearGroup make_fp32_quant_group(
    Collectives& collectives,
    int primary,
    std::span<const ::tyloi::Tensor> weights);

QuantLinearGroup make_quant_group(
    std::vector<::tyloi::Tensor> layers,
    bool group_nint,
    bool cpu_layer,
    bool preserve_projection_boundaries = false);

DenseLinearGroup make_dense_group(
    const std::vector<::tyloi::Tensor>& weights);

// Bound resources and options must outlive this object; drain before resetting them.
class Projector {
public:
    struct Options {
        bool peer_first = true;
        bool reduce_to_primary = true;
        bool fp16_reduce = true;
        bool grouped = true;
        bool nint_grouped = true;
        bool parallel = true;
        bool nvq_fusion = true;
        MoeOptions moe;
    };

    Projector(CudaProfiler& profiler, KlMmqState& mmq, TqExec& tq,
              Branches& branches, Collectives& collectives, MfeScratch& mfe_scratch,
              MoeScratchCache& moe_scratch, const Options& options,
              int primary, const bool& serial, const bool& projection_major, bool independent)
        : profiler_(profiler), mmq_(mmq), tq_(tq), branches_(branches),
          collectives_(collectives), mfe_scratch_(mfe_scratch), moe_scratch_(moe_scratch), options_(options), primary_(primary),
          serial_(serial), projection_major_(projection_major), independent_(independent) {}

    int primary_device() const { return primary_; }
    QuantLinearGroup group(std::span<const ::tyloi::Tensor> weights) const;
    bool grouped(std::span<const ::tyloi::Tensor> weights) const;
    MoeScratchCache& moe_scratch() const { return moe_scratch_; }

    ::tyloi::DenseTensor run(const ::tyloi::Tensor& weight, ::tyloi::DenseTensor input) const;
    ::tyloi::DenseTensor run(const ::tyloi::Tensor& weight, ::tyloi::DenseTensor input,
        ::tyloi::DenseTensor ids) const;
    ::tyloi::DenseTensor run(const ::tyloi::Tensor& weight, ::tyloi::DenseTensor input,
        const MoeRoutePlan& route) const;
    std::vector<::tyloi::DenseTensor> run(
        std::span<const ::tyloi::Tensor> weights,
        ::tyloi::DenseTensor input, size_t prefix = 1) const;
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
        const std::vector<::tyloi::Tensor>& projections, ::tyloi::DenseTensor input) const;

    ::tyloi::DenseTensor run(const ::tyloi::MfeTensor& weight, ::tyloi::DenseTensor input,
        const MoeRoutePlan& route, bool prequantized = false, int epilogue = 0) const;
    ::tyloi::DenseTensor glu(const ::tyloi::MfeTensor& weight, ::tyloi::DenseTensor input,
        const MoeRoutePlan& route, bool gelu) const;
    ::tyloi::DenseTensor gated(const ::tyloi::MfeTensor& weight, ::tyloi::DenseTensor input,
        const MoeRoutePlan& route, bool gelu) const;
    ::tyloi::DenseTensor clamped(const ::tyloi::MfeTensor& weight, ::tyloi::DenseTensor input,
        const MoeRoutePlan& route, double limit) const;
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
    Options options_;
    int primary_;
    const bool& serial_;
    const bool& projection_major_;
    bool independent_;

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
