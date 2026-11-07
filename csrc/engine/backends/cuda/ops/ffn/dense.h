#pragma once

#include "mlp.h"

#include <optional>

namespace tyloi::cuda {

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
#ifdef TYLOINFER_NATIVE_CUDA
    const ::tyloi::Tensor& norm;
#else
    ::tyloi::DenseTensor norm;
#endif
    double eps, offset;
    bool bf16 = false;
    bool trace = false;
    bool linear = false;

    ::tyloi::DenseTensor forward(
        const Projector& project, CudaProfiler& profiler, KlMmqState& mmq, TqExec& tq,
        Branches& branches, Collectives& collectives, const Projector::Options& parallel,
        const Options& options, bool serial, ::tyloi::DenseTensor residual,
        ::tyloi::DenseTensor input, Trace* trace_output = nullptr) const;
};

} // namespace tyloi::cuda
