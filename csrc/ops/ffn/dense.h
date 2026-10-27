#pragma once

#include "ops/ffn/mlp.h"

#include <optional>

namespace mfq::cuda {

class Branches;

struct Dense {
    struct Options {
        Mlp::Options mlp;
        bool important_parallel = true;
        bool important_fp32 = false;
        bool fp32_residual = false;
        bool bf16_norm = true;
        bool bf16_swiglu = true;
        bool bf16_residual = true;
    };

    Mlp mlp;
    std::optional<Mlp> important;
    mfq_tensor_backend::Tensor norm;
    double eps, offset;
    bool bf16 = false;
    bool trace = false;
    bool linear = false;

    mfq_tensor_backend::Tensor forward(
        const Projector& project, CudaProfiler& profiler, KlMmqState& mmq, TqExec& tq,
        Branches& branches, Collectives& collectives, const Projector::Options& parallel,
        const Options& options, bool serial, mfq_tensor_backend::Tensor residual,
        mfq_tensor_backend::Tensor input, Trace* trace_output = nullptr) const;
};

} // namespace mfq::cuda
