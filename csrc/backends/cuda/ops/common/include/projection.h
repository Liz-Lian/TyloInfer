#pragma once

#include "quant_linear.h"
#include "moe.h"

struct TqMoeBanks;

namespace mfq::cuda {

class Branches;
struct Experts;

// Borrowed for one forward; the owner drains before resetting its resources.
class Projector {
public:
    struct Options {
        int primary = 0;
        bool peer_first = true;
        bool reduce_to_primary = true;
        bool fp16_reduce = true;
        bool grouped = true;
        bool parallel = true;
        bool nvq_fusion = true;
        bool nint_group_quantize = true;
        MoeOptions moe;
    };

    Projector(CudaProfiler& profiler, KlMmqState& mmq, TqExec& tq,
              Branches& branches, Collectives& collectives,
              const bool& serial, const bool& projection_major, Options options)
        : profiler_(profiler), mmq_(mmq), tq_(tq), branches_(branches),
          collectives_(collectives), serial_(serial),
          projection_major_(projection_major), options_(options) {}

    mfq_tensor_backend::Tensor run(const QuantLinear& weight, mfq_tensor_backend::Tensor input) const;
    std::vector<mfq_tensor_backend::Tensor> run(
        const QuantLinearGroup& group, mfq_tensor_backend::Tensor input) const;
    mfq_tensor_backend::Tensor gated(const QuantLinear& weight, mfq_tensor_backend::Tensor input,
                                    mfq_tensor_backend::Tensor gate, InputGateMode mode) const;
    mfq_tensor_backend::Tensor groupwise(const QuantLinear& weight,
                                       mfq_tensor_backend::Tensor input, int64_t groups) const;
    mfq_tensor_backend::Tensor swiglu(const QuantLinearGroup& group, mfq_tensor_backend::Tensor input) const;
    mfq_tensor_backend::Tensor geglu(const QuantLinearGroup& group, mfq_tensor_backend::Tensor input) const;
    std::vector<mfq_tensor_backend::Tensor> grouped(
        const QuantLinearProjectionRefs& projections, mfq_tensor_backend::Tensor input) const;

    mfq_tensor_backend::Tensor run(
        const Experts& weight, mfq_tensor_backend::Tensor input, const MoeRoutePlan& route,
        bool prequantized = false, int epilogue = 0) const;
    mfq_tensor_backend::Tensor glu(
        const Experts& weight, mfq_tensor_backend::Tensor input, const MoeRoutePlan& route,
        bool gelu) const;
    mfq_tensor_backend::Tensor gated(
        const Experts& weight, mfq_tensor_backend::Tensor input, const MoeRoutePlan& route,
        bool gelu) const;
    mfq_tensor_backend::Tensor clamped(
        const Experts& weight, mfq_tensor_backend::Tensor input, const MoeRoutePlan& route,
        double limit) const;
    mfq_tensor_backend::Tensor run(
        const TqMoeBanks& banks, mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor ids, mfq_tensor_backend::Tensor weights,
        float swiglu_limit = 0) const;

private:
    CudaProfiler& profiler_;
    KlMmqState& mmq_;
    TqExec& tq_;
    Branches& branches_;
    Collectives& collectives_;
    const bool& serial_;
    const bool& projection_major_;
    Options options_;

    mfq_tensor_backend::Tensor sharded(const QuantLinear& weight, mfq_tensor_backend::Tensor input,
                                      MfqOptional<mfq_tensor_backend::Tensor> gate, InputGateMode mode) const;
    std::vector<mfq_tensor_backend::Tensor> run(
        const NintLinearGroup& group, mfq_tensor_backend::Tensor input) const;
    mfq_tensor_backend::Tensor swiglu(const NintLinearGroup& group, mfq_tensor_backend::Tensor input) const;
    mfq_tensor_backend::Tensor geglu(const NintLinearGroup& group, mfq_tensor_backend::Tensor input) const;
};

} // namespace mfq::cuda
