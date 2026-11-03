#pragma once

#include "engine/backends/cuda/executor/task.h"
#include "engine/backends/cuda/executor/moe_ffn_pipeline.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/executor.h"
#endif

#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <tuple>

class TqExec;

namespace mfq::cuda {

class Branches;
class Collectives;
class Pipelines;
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

#endif
    Collectives& collectives() noexcept { return collectives_; }
    const Collectives& collectives() const noexcept { return collectives_; }
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
        Ffn::Options ffn;
        MoeFfnOptions moe_ffn;
        std::string profile_filter;
    };

    static void prepare(std::int64_t vocabulary);

    Executor(Options options, int primary);
    ~Executor();
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    std::unique_ptr<Executor> fork() const;
    std::unique_ptr<Executor> fork(int primary) const;
    ContextSet& contexts() noexcept { return contexts_; }
    const ContextSet& contexts() const noexcept { return contexts_; }

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
    KlMmqState& mmq() noexcept { return mmq_; }
    const KlMmqState& mmq() const noexcept { return mmq_; }
    const std::shared_ptr<TqExec>& tq() const noexcept { return tq_; }
    RouteStats& routes() noexcept { return routes_; }
    const RouteStats& routes() const noexcept { return routes_; }
    Branches& branches() noexcept { return branches_; }
    Pipelines& pipelines() noexcept { return pipelines_; }
    const Options& options() const noexcept { return options_; }
    const AttentionOptions& attention_options() const noexcept { return options_.attention; }
    bool fp32_residual() const noexcept { return options_.dense.fp32_residual; }
    AttentionScratch& attention_scratch(std::size_t index);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    using AttentionGeometry = std::tuple<std::int64_t, std::int64_t, std::int64_t, std::int64_t>;
    AttentionScratch& attention_scratch(mfq::Device device, AttentionGeometry geometry);
#endif
    MlaScratch& mla_scratch(int device) { return mla_scratch_[device]; }

private:
    Executor(ContextSet& contexts, CudaProfiler& profiler, KlMmqState& mmq, const std::shared_ptr<TqExec>& tq,
             Branches& branches, Pipelines& pipelines, Collectives& collectives, RouteStats& routes,
             const Options& options, int primary);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    std::unique_ptr<mfq::Executor::Scope> enter(int device);
#endif
    friend class Ffn;
    std::unique_ptr<ContextSet> owned_contexts_;
    ContextSet& contexts_;
    std::unique_ptr<Options> owned_options_;
    const Options& options_;
    std::unique_ptr<CudaProfiler> owned_profiler_;
    CudaProfiler& profiler_;
    std::unique_ptr<KlMmqState> owned_mmq_;
    KlMmqState& mmq_;
    std::shared_ptr<TqExec> tq_;
    std::unique_ptr<Branches> owned_branches_;
    Branches& branches_;
    std::unique_ptr<Collectives> owned_collectives_;
    Collectives& collectives_;
    std::unique_ptr<RouteStats> owned_routes_;
    RouteStats& routes_;
    std::unique_ptr<Pipelines> owned_pipelines_;
    Pipelines& pipelines_;
    int primary_;
    std::vector<AttentionScratch> scratch_;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    mutable std::mutex attention_scratch_mutex_;
    std::map<std::pair<int, std::uintptr_t>, std::map<AttentionGeometry, AttentionScratch>> attention_scratch_;
#endif
    std::map<int, MlaScratch> mla_scratch_;
};

} // namespace mfq::cuda
