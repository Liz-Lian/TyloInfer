#pragma once

#include "engine/models/ffn.h"
#include "native/cuda/branches.h"
#include "native/cuda/collectives.h"
#include "native/cuda/profile.h"
#include "diagnostics/route_stats.h"
#include "ops/attention/full.h"
#include "ops/attention/linear.h"
#include "ops/common/include/mmq.h"
#include "ops/common/include/moe.h"
#include "ops/common/include/projection.h"
#include "ops/ffn/dense.h"

#include <optional>

class TqExec;

namespace mfq::cuda {

// Borrowed resources and options outlive this view; Mode is scoped to one call.
class FfnExec {
public:
    struct Options {
        bool small_heterogeneous = true;
        bool delayed_readback = true;
        bool bundle_prefetch = true;
        bool swiglu_fusion = true;
        bool reduce_fusion = true;
    };

    struct Mode {
        bool materialized = false;
        bool unfused = false;
        bool serial = false;
        bool batch_serial = false;
        int tile = 0;
        int tiles = 1;
        bool reuse_input = true;
    };

    FfnExec(Projector project, CudaProfiler& profiler, KlMmqState& mmq,
            TqExec& tq, Branches& branches, Collectives& collectives,
            RouteStats& stats, const Projector::Options& projection,
            const FullAttention::Options& attention, const Dense::Options& dense,
            const Options& options, Mode mode);
    FfnExec(const FfnExec&) = delete;
    FfnExec& operator=(const FfnExec&) = delete;

    mfq_tensor_backend::Tensor run(
        const Ffn& ffn, mfq_tensor_backend::Tensor input,
        MfqOptional<mfq_tensor_backend::Tensor> input_ids = mfq_nullopt,
        MfqOptional<mfq_tensor_backend::Tensor> router_bias = mfq_nullopt);
    mfq_tensor_backend::Tensor full(
        const Ffn& ffn, const mfq_tensor_backend::Tensor& ffn_norm,
        const FullAttention& attention, mfq_tensor_backend::Tensor residual,
        mfq_tensor_backend::Tensor output, Trace* trace = nullptr);
    mfq_tensor_backend::Tensor linear(
        const Ffn& ffn, const mfq_tensor_backend::Tensor& ffn_norm,
        const LinearAttention& attention, mfq_tensor_backend::Tensor residual,
        mfq_tensor_backend::Tensor output);
    void prefetch(
        const Ffn& ffn, const mfq_tensor_backend::Tensor& input, bool enabled,
        MfqOptional<mfq_tensor_backend::Tensor> input_ids = mfq_nullopt);

private:
    mfq_tensor_backend::Tensor run_impl(
        const Ffn& ffn, mfq_tensor_backend::Tensor input,
        MfqOptional<mfq_tensor_backend::Tensor> input_ids,
        bool allow_important_neurons,
        MfqOptional<mfq_tensor_backend::Tensor> router_bias);
    mfq_tensor_backend::Tensor run_experts(
        const Ffn& ffn, mfq_tensor_backend::Tensor input,
        const MoeRoutePlan& route,
        mfq_tensor_backend::Tensor route_weights);

    Projector project_;
    CudaProfiler& profiler_;
    KlMmqState& mmq_;
    TqExec& tq_;
    Branches& branches_;
    Collectives& collectives_;
    RouteStats& stats_;
    const Projector::Options& projection_;
    const FullAttention::Options& attention_;
    const Dense::Options& dense_;
    const Options& options_;
    Mode mode_;
};

bool uses_expert_cache(const Ffn& ffn);
bool expert_parallel(const Ffn& ffn);

} // namespace mfq::cuda
