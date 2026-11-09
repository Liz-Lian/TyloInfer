#pragma once

#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/ops/attention/attention.h"
#include "engine/backends/cuda/ops/attention/linear.h"
#include "engine/backends/cuda/ops/ffn/dense.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/backends/cuda/ops/ffn/gated.h"
#include "engine/executor.h"

#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

class TqExec;

namespace tyloi::cuda {

class Branches;
class Collectives;
class RouteStats;
struct FullAttention;

class Executor
#ifdef TYLOINFER_NATIVE_CUDA
    : public tyloi::Executor
#endif
{
public:
#ifdef TYLOINFER_NATIVE_CUDA
    std::unique_ptr<tyloi::Executor::Scope> enter() override;
    std::unique_ptr<tyloi::Executor::Timer> timer() override;
    std::unique_ptr<tyloi::Executor::Profile> profile(const char* name) override;
    std::unique_ptr<tyloi::Executor::Graph> graph() override;
    std::vector<tyloi::Executor::Memory> memory(std::span<const tyloi::Device> devices) const override;
    void drain() override;
    void clear() override;
    // Set the default device during loading, before publication or concurrent submissions.
    void set_primary(int primary) noexcept { primary_ = primary; }
#else
    std::unique_ptr<tyloi::Executor::Scope> enter();
    std::vector<tyloi::Executor::Memory> memory(std::span<const tyloi::Device> devices) const;
#endif
    std::unique_ptr<tyloi::Executor::Scope> enter(int device);
    void join();
    static Executor& current();
#ifdef TYLOINFER_NATIVE_CUDA
    void wait();
    static const Ffn::Mode& current_ffn();
    std::unique_ptr<tyloi::Executor::Scope> enter(int device, const Ffn::Mode& mode);
    // Override only the policy on this active executor; retain its device and stream.
    // Nested policy scopes retain an outer serial-branch requirement.
    std::unique_ptr<tyloi::Executor::Scope> enter(const Ffn::Mode& mode);
#else
    void wait(const std::unordered_set<int>& devices);
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
        GatedResidual::Options residual;
        bool gdn_fused_output = true;
        bool gdn_fused_preparation = true;
        bool gdn_fused_core = true;
        bool tensor_parallel_shared_linear_attention_input = true;
        bool linear_conv_prefill_fused = true;
        bool gdn_transposed_state = true;
        bool gemma4_fused_norms = true;
        bool dsv4_groupwise_output_a = true;
        bool report_memory = false;
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

#ifdef TYLOINFER_NATIVE_CUDA
    std::vector<std::shared_ptr<const ::tyloi::Tensor>> storage() const;
#else
    std::vector<::tyloi::DenseTensor> storage() const;
#endif
    Projector projector(const Ffn::Mode& mode);
    Projector projector(const Ffn::Mode& mode, int primary);
    ::tyloi::DenseTensor full_ffn(
        const Ffn& ffn,
#ifdef TYLOINFER_NATIVE_CUDA
        const ::tyloi::Tensor& ffn_norm,
#else
        const ::tyloi::DenseTensor& ffn_norm,
#endif
        const FullAttention& attention, ::tyloi::DenseTensor residual,
        ::tyloi::DenseTensor output, const Ffn::Mode& mode, Trace* trace = nullptr);
    ::tyloi::DenseTensor linear_ffn(
        const Ffn& ffn,
#ifdef TYLOINFER_NATIVE_CUDA
        const ::tyloi::Tensor& ffn_norm,
#else
        const ::tyloi::DenseTensor& ffn_norm,
#endif
        const LinearAttention& attention, ::tyloi::DenseTensor residual,
        ::tyloi::DenseTensor output, const Ffn::Mode& mode);
    CudaProfiler& profiler() noexcept { return profiler_; }
    KlMmqState& mmq() noexcept { return mmq_; }
    const KlMmqState& mmq() const noexcept { return mmq_; }
    const std::shared_ptr<TqExec>& tq() const noexcept { return tq_; }
    RouteStats& routes() noexcept { return routes_; }
    const RouteStats& routes() const noexcept { return routes_; }
    Branches& branches() noexcept { return branches_; }
    const Options& options() const noexcept { return options_; }
    Mode& mode() noexcept { return mode_; }
    const Mode& mode() const noexcept { return mode_; }
    Mode forward_mode(int primary) const;
    const AttentionOptions& attention_options() const noexcept { return options_.attention; }
    bool fp32_residual() const noexcept { return options_.dense.fp32_residual; }
    AttentionScratch& attention_scratch(std::size_t index);
#ifdef TYLOINFER_NATIVE_CUDA
    using AttentionGeometry = std::tuple<std::int64_t, std::int64_t, std::int64_t, std::int64_t>;
    AttentionScratch& attention_scratch(tyloi::Device device, AttentionGeometry geometry);
#else
    std::unordered_map<const void*, AttentionScratch>& attention_scratch() noexcept { return attention_scratch_; }
#endif
    MlaScratch& mla_scratch(int device) { return mla_scratch_[device]; }

private:
    Executor(ContextSet& contexts, CudaProfiler& profiler, KlMmqState& mmq, const std::shared_ptr<TqExec>& tq,
             Branches& branches, Collectives& collectives, RouteStats& routes,
             const Options& options, Mode& mode, int primary);
    friend class Ffn;
    std::unique_ptr<ContextSet> owned_contexts_;
    ContextSet& contexts_;
    std::unique_ptr<Options> owned_options_;
    const Options& options_;
    std::unique_ptr<Mode> owned_mode_;
    Mode& mode_;
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
    int primary_;
    std::vector<AttentionScratch> scratch_;
    MfeScratch mfe_scratch_;
    MoeScratchCache moe_scratch_;
#ifdef TYLOINFER_NATIVE_CUDA
    mutable std::mutex attention_scratch_mutex_;
    std::map<std::pair<int, std::uintptr_t>, std::map<AttentionGeometry, AttentionScratch>> attention_scratch_;
#else
    std::unordered_map<const void*, AttentionScratch> attention_scratch_;
#endif
    std::map<int, MlaScratch> mla_scratch_;
};

} // namespace tyloi::cuda
