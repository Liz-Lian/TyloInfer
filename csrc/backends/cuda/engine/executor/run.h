#pragma once

#include "engine/task.h"
#include "engine/executor/ffn.h"

namespace mfq::cuda {

class Branches;
class Collectives;

class Executor {
public:
    struct Options {
        Projector::Options projection;
        FullAttention::Options attention{
            .kernels = {.bf16_flash128 = true, .flash128_casts = true,
                .bf16_gqa_decode = true, .decode_aten = false,
                .mma_prefill = true, .mma_decode = true, .split_k = true}};
        Dense::Options dense;
    };

    Executor(CudaProfiler& profiler, KlMmqState& mmq, TqExec& tq,
             Branches& branches, Collectives& collectives, RouteStats& routes,
             const Options& options, const FfnExec::Options& ffn_options,
             int primary, const bool& serial, const bool& projection_major)
        : profiler_(profiler), mmq_(mmq), tq_(tq), branches_(branches), collectives_(collectives),
          routes_(routes), options_(options), primary_(primary), serial_(serial),
          projection_major_(projection_major), ffn_options_(ffn_options) {}
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    mfq_tensor_backend::Tensor run(std::span<const Task> tasks, Inputs inputs,
                                   FfnExec::Mode mode = {});
    std::span<const AttentionScratch> scratch() const noexcept { return scratch_; }

private:
    CudaProfiler& profiler_;
    KlMmqState& mmq_;
    TqExec& tq_;
    Branches& branches_;
    Collectives& collectives_;
    RouteStats& routes_;
    const Options& options_;
    int primary_;
    const bool& serial_;
    const bool& projection_major_;
    const FfnExec::Options& ffn_options_;
    std::vector<AttentionScratch> scratch_;
};

} // namespace mfq::cuda
