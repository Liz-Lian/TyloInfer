#pragma once

#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/ops/attention/attention.h"
#include "engine/backends/cuda/ops/attention/linear.h"
#include "engine/backends/cuda/ops/ffn/dense.h"
#include "engine/backends/cuda/ops/ffn/options.h"
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
namespace tyloi::weights { class Cache; }

namespace tyloi::cuda {

class Branches;
class Collectives;
class RouteStats;
class KvMemoryManager;
class KvOffloadStore;
struct KvMemoryAllocation;
class DecodeWindow;

class Executor
#ifdef TYLOINFER_NATIVE_CUDA
    : public tyloi::Executor
#endif
{
public:
#ifdef TYLOINFER_NATIVE_CUDA
    /** Returns the primary CUDA device selected before publication. */
    tyloi::Device device() const noexcept override { return tyloi::Device(tyloi::kCUDA, primary_); }
    std::unique_ptr<tyloi::Executor::Scope> enter() override;
    std::unique_ptr<tyloi::Executor::Profile> profile(const char* name) override;
    std::unique_ptr<tyloi::Executor::Graph> graph() override;
    /** Query device memory snapshots within this executor's scope. */
    std::vector<tyloi::Executor::Memory> memory(std::span<const tyloi::Device> devices) const override;
    /** Report the configured KV manager and offload store metrics. */
    tyloi::engine::Metrics metrics() const override;
    /** Reclaim idle allocator memory on the device. */
    void trim(tyloi::Device device) override;
    /** Set the device allocation limit; zero removes it. */
    void limit(tyloi::Device device, std::size_t bytes) override;
    /** Bind the budget used for KV admission and offload. */
    void bind_budget(tyloi::engine::Budget* budget) noexcept override;
    /** Install load-time KV owners before creating request executors. */
    void configure_kv(std::shared_ptr<KvMemoryManager> kv, std::shared_ptr<KvOffloadStore> offload);
    void drain() override;
    void clear() override;
    // Set the default device during loading, before publication or concurrent submissions.
    void set_primary(int primary) noexcept { primary_ = primary; }
#else
    std::unique_ptr<tyloi::Executor::Scope> enter();
    /** Query device memory snapshots within this executor's scope. */
    std::vector<tyloi::Executor::Memory> memory(std::span<const tyloi::Device> devices) const;
    /** Reclaim idle allocator memory on the device. */
    void trim(tyloi::Device device);
    /** Set the device allocation limit; zero removes it. */
    void limit(tyloi::Device device, std::size_t bytes);
#endif
    /** Start a CUDA event timer for the current execution stream. */
    std::unique_ptr<tyloi::Executor::Timer> timer()
#ifdef TYLOINFER_NATIVE_CUDA
        override
#endif
        ;
    std::unique_ptr<tyloi::Executor::Scope> enter(int device);
    void join();
    static Executor& current();
#ifdef TYLOINFER_NATIVE_CUDA
    void wait();
    static const FfnMode& current_ffn();
    std::unique_ptr<tyloi::Executor::Scope> enter(int device, const FfnMode& mode);
    // Override only the policy on this active executor; retain its device and stream.
    // Nested policy scopes retain an outer serial-branch requirement.
    std::unique_ptr<tyloi::Executor::Scope> enter(const FfnMode& mode);
#else
    void wait(const std::unordered_set<int>& devices);
#endif
    Collectives& collectives() noexcept { return collectives_; }
    const Collectives& collectives() const noexcept { return collectives_; }
    struct Mode {
        FfnMode ffn;
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
        DenseOptions dense;
        FfnOptions ffn;
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
    /** Reset execution resources for loading after callers drain work and retire graphs. */
    void reset();
    std::unique_ptr<Executor> fork() const;
    std::unique_ptr<Executor> fork(int primary) const;
    ContextSet& contexts() noexcept { return contexts_; }
    const ContextSet& contexts() const noexcept { return contexts_; }

#ifdef TYLOINFER_NATIVE_CUDA
    std::vector<::tyloi::Tensor> storage() const;
#else
    std::vector<::tyloi::DenseTensor> storage() const;
#endif
    Projector projector(const FfnMode& mode);
    Projector projector(const FfnMode& mode, int primary);
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
#ifdef TYLOINFER_NATIVE_CUDA
    friend class ::tyloi::weights::Cache;
    friend class DecodeWindow;
    static Executor* active() noexcept;
    friend class KvMemoryManager;
    friend std::shared_ptr<KvMemoryAllocation> allocate_kv(int64_t, int64_t, int64_t, int64_t,
        int64_t, int64_t, const ::tyloi::TensorOptions&, int64_t);
#endif
    Executor(ContextSet& contexts, CudaProfiler& profiler, KlMmqState& mmq, const std::shared_ptr<TqExec>& tq,
             Branches& branches, Collectives& collectives, RouteStats& routes,
             const Options& options, Mode& mode, int primary);
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
    ::tyloi::weights::Cache* weights_ = nullptr;
    std::shared_ptr<KvMemoryManager> kv_;
    std::shared_ptr<KvOffloadStore> offload_;
    mutable std::mutex attention_scratch_mutex_;
    std::map<std::pair<int, std::uintptr_t>, std::map<AttentionGeometry, AttentionScratch>> attention_scratch_;
#else
    std::unordered_map<const void*, AttentionScratch> attention_scratch_;
#endif
    std::map<int, MlaScratch> mla_scratch_;
};

} // namespace tyloi::cuda
