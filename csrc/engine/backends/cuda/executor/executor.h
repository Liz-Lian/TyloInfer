#pragma once

#include "engine/backends/cuda/executor/task.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/executor.h"
#endif

#include <memory>
#include <map>
#include <mutex>
#include <tuple>

namespace mfq::cuda {

class Branches;
class Collectives;
class RouteStats;

class Executor
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    : public mfq::Executor
#endif
{
public:
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    std::unique_ptr<mfq::Executor::Scope> enter() override;
    static Executor& current();

    Collectives& collectives() noexcept { return collectives_; }
#endif
    struct Mode {
        Ffn::Mode ffn;
        LinearAttention::Options linear;
        std::int64_t kv_capacity = 0;
        bool transposed = true;
        Trace* trace = nullptr;
        int trace_layer = -1;
        int primary = 0;
        bool gemma4_fused_norms = true;
        bool groupwise_output = true;
    };

    struct Options {
        Projector::Options projection;
        AttentionOptions attention{
            .kernels = {.bf16_flash128 = true, .flash128_casts = true,
                .bf16_gqa_decode = true, .decode_aten = false,
                .mma_prefill = true, .mma_decode = true, .split_k = true}};
        Dense::Options dense;
    };

    static void prepare(std::int64_t vocabulary);

    Executor(ContextSet& contexts, CudaProfiler& profiler, KlMmqState& mmq, TqExec& tq,
             Branches& branches, Collectives& collectives, RouteStats& routes,
             const Options& options, const Ffn::Options& ffn_options,
             int primary)
        : contexts_(contexts), profiler_(profiler), mmq_(mmq), tq_(tq), branches_(branches), collectives_(collectives),
          routes_(routes), options_(options), primary_(primary), ffn_options_(ffn_options) {}
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    std::unique_ptr<Executor> fork() const;

    mfq_tensor_backend::Tensor run(std::span<const Task> tasks, Inputs inputs,
                                   const Mode& mode);
    std::vector<mfq_tensor_backend::Tensor> storage() const;
    Projector projector(const Ffn::Mode& mode);
    mfq_tensor_backend::Tensor full_ffn(
        const Ffn& ffn, const mfq_tensor_backend::Tensor& ffn_norm,
        const FullAttention& attention, mfq_tensor_backend::Tensor residual,
        mfq_tensor_backend::Tensor output, const Ffn::Mode& mode, Trace* trace = nullptr);
    mfq_tensor_backend::Tensor linear_ffn(
        const Ffn& ffn, const mfq_tensor_backend::Tensor& ffn_norm,
        const LinearAttention& attention, mfq_tensor_backend::Tensor residual,
        mfq_tensor_backend::Tensor output, const Ffn::Mode& mode);
    CudaProfiler& profiler() noexcept { return profiler_; }
    const AttentionOptions& attention_options() const noexcept { return options_.attention; }
    bool fp32_residual() const noexcept { return options_.dense.fp32_residual; }
    AttentionScratch& attention_scratch(std::size_t index);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    using AttentionGeometry = std::tuple<std::int64_t, std::int64_t, std::int64_t, std::int64_t>;
    AttentionScratch& attention_scratch(mfq::Device device, AttentionGeometry geometry);
#endif
    MlaScratch& mla_scratch(int device) { return mla_scratch_[device]; }

private:
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    std::unique_ptr<mfq::Executor::Scope> enter(int device);
#endif
    friend class Ffn;
    ContextSet& contexts_;
    CudaProfiler& profiler_;
    KlMmqState& mmq_;
    TqExec& tq_;
    Branches& branches_;
    Collectives& collectives_;
    RouteStats& routes_;
    const Options& options_;
    int primary_;
    const Ffn::Options& ffn_options_;
    std::vector<AttentionScratch> scratch_;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    mutable std::mutex attention_scratch_mutex_;
    std::map<std::pair<int, std::uintptr_t>, std::map<AttentionGeometry, AttentionScratch>> attention_scratch_;
#endif
    std::map<int, MlaScratch> mla_scratch_;
};

} // namespace mfq::cuda
