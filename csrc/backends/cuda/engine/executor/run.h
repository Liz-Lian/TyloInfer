#pragma once

#include "engine/task.h"
#include "engine/executor/ffn.h"
#include "engine/executor/input.h"

#include <memory>

namespace mfq::cuda {

class Branches;
class Collectives;

class Executor {
public:
    struct Mode {
        FfnExec::Mode ffn;
        LinearAttention::Options linear;
        std::int64_t kv_capacity = 0;
        bool transposed = true;
        Trace* trace = nullptr;
        int trace_layer = -1;
        int primary = 0;
        bool projection_major = false;
    };

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
             int primary)
        : profiler_(profiler), mmq_(mmq), tq_(tq), branches_(branches), collectives_(collectives),
          routes_(routes), options_(options), primary_(primary), ffn_options_(ffn_options) {}
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    std::unique_ptr<Executor> fork() const;

    mfq_tensor_backend::Tensor run(std::span<const Task> tasks, Inputs inputs,
                                   const Mode& mode);
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
    const FfnExec::Options& ffn_options_;
    std::vector<AttentionScratch> scratch_;
};

} // namespace mfq::cuda
