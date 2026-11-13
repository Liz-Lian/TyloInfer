#pragma once

#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/ops/attention/attention.h"
#include "engine/backends/cuda/ops/attention/linear.h"
#include "engine/backends/cuda/ops/ffn/dense.h"
#include "engine/backends/cuda/ops/ffn/options.h"
#include "engine/executor.h"

#include <memory>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
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
class Graph;

/** Optional scoped overrides for native model computation choices. */
struct ComputePolicy {
    std::optional<bool> rotary;
    std::optional<bool> attention_grouped;
    std::optional<bool> residual_two_stage;
    std::optional<bool> residual_dense_injection;
};

class Executor
    : public tyloi::Executor
{
public:
    /** Returns the primary CUDA device selected before publication. */
    tyloi::Device device() const noexcept override { return tyloi::Device(tyloi::kCUDA, primary_); }
    std::unique_ptr<tyloi::Executor::Scope> enter() override;
    /** Create the working set for one prefill job. */
    std::unique_ptr<tyloi::Executor::Work> work() override;
    std::unique_ptr<tyloi::Executor::Profile> profile(const char* name) override;
    std::unique_ptr<tyloi::Executor::Graph> graph() override;
    /** Query device memory snapshots within this executor's scope. */
    std::vector<tyloi::Executor::Memory> memory(std::span<const tyloi::Device> devices,
        tyloi::Executor::MemoryDetail detail = tyloi::Executor::MemoryDetail::basic) const override;
    /** Wait for the current CUDA execution stream. */
    void synchronize() override;
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
    /** Start a CUDA event timer for the current execution stream. */
    std::unique_ptr<tyloi::Executor::Timer> timer()
        override
        ;
    std::unique_ptr<tyloi::Executor::Scope> enter(int device);
    void join();
    static Executor& current();
    void wait();
    static const FfnMode& current_ffn();
    std::unique_ptr<tyloi::Executor::Scope> enter(int device, const FfnMode& mode);
    // Override only the policy on this active executor; retain its device and stream.
    // Nested policy scopes retain an outer serial-branch requirement.
    std::unique_ptr<tyloi::Executor::Scope> enter(const FfnMode& mode);
    Collectives& collectives() noexcept { return collectives_; }
    const Collectives& collectives() const noexcept { return collectives_; }
    struct Mode {
        FfnMode ffn;
        LinearCore::Options linear;
        std::int64_t kv_capacity = 0;
        bool transposed = true;
        Trace* trace = nullptr;
        int trace_layer = -1;
        int primary = 0;
        bool gemma4_fused_norms = true;
        bool groupwise_output = true;
        /** Optional diagnostic kernel policy inherited by executor forks. */
        std::optional<AttentionKernels> attention_kernels;
        ComputePolicy compute;
    };

    struct Options {
        Projector::Options projection;
        AttentionOptions attention{
            .kernels = {.bf16_flash128 = true, .flash128_casts = true,
                .bf16_gqa_decode = true, .decode_aten = false,
                .mma_prefill = true, .mma_decode = true, .split_k = true}};
        DenseOptions dense;
        FfnOptions ffn;
        ResidualOptions residual;
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

    std::vector<::tyloi::Tensor> storage() const;
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
    /** Return residual settings with the current scoped overrides applied. */
    ResidualOptions residual_options() const {
        auto result = options_.residual;
        if (mode_.compute.residual_two_stage) result.two_stage = *mode_.compute.residual_two_stage;
        if (mode_.compute.residual_dense_injection) result.dense_injection = *mode_.compute.residual_dense_injection;
        return result;
    }
    Mode forward_mode(int primary) const;
    const AttentionOptions& attention_options() const noexcept { return options_.attention; }
    /** Return the active attention kernel policy, including any scoped override. */
    const AttentionKernels& attention_kernels() const noexcept {
        return mode_.attention_kernels ? *mode_.attention_kernels : options_.attention.kernels;
    }
    bool fp32_residual() const noexcept { return options_.dense.fp32_residual; }
    AttentionScratch& attention_scratch(std::size_t index);
    using AttentionGeometry = std::tuple<std::int64_t, std::int64_t, std::int64_t, std::int64_t>;
    AttentionScratch& attention_scratch(tyloi::Device device, AttentionGeometry geometry);
    MlaScratch& mla_scratch(int device) { return mla_scratch_[device]; }

private:
    friend class ::tyloi::weights::Cache;
    friend class DecodeWindow;
    friend class RotaryEmbedding;
    static Executor* active() noexcept;
    friend class KvMemoryManager;
    friend std::shared_ptr<KvMemoryAllocation> allocate_kv(int64_t, int64_t, int64_t, int64_t,
        int64_t, int64_t, const ::tyloi::TensorOptions&, int64_t);
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
    ::tyloi::weights::Cache* weights_ = nullptr;
    std::map<std::pair<int, cudaStream_t>, std::weak_ptr<::tyloi::cuda::Graph>> window_pools_;
    std::shared_ptr<KvMemoryManager> kv_;
    std::shared_ptr<KvOffloadStore> offload_;
    mutable std::mutex attention_scratch_mutex_;
    std::map<std::pair<int, std::uintptr_t>, std::map<AttentionGeometry, AttentionScratch>> attention_scratch_;
    std::map<int, MlaScratch> mla_scratch_;
};

/** Restore computation choices after a quiescent scoped diagnostic operation. */
class PolicyScope {
public:
    PolicyScope(Executor& executor, ComputePolicy policy)
        : executor_(executor), previous_(executor.mode().compute) {
        if (&Executor::current() != &executor)
            throw std::logic_error("compute policy requires the active executor");
        auto selected = previous_;
        if (policy.rotary) selected.rotary = policy.rotary;
        if (policy.attention_grouped) selected.attention_grouped = policy.attention_grouped;
        if (policy.residual_two_stage) selected.residual_two_stage = policy.residual_two_stage;
        if (policy.residual_dense_injection) selected.residual_dense_injection = policy.residual_dense_injection;
        executor_.mode().compute = std::move(selected);
    }
    ~PolicyScope() { executor_.mode().compute = std::move(previous_); }
    PolicyScope(const PolicyScope&) = delete;
    PolicyScope& operator=(const PolicyScope&) = delete;
private:
    Executor& executor_;
    ComputePolicy previous_;
};

/** Temporarily override only attention kernel choices on the active executor. */
class AttentionScope {
public:
    AttentionScope(Executor& executor, AttentionKernels kernels)
        : executor_(executor), previous_(executor.mode().attention_kernels) {
        if (&Executor::current() != &executor)
            throw std::logic_error("attention policy requires the active executor");
        executor_.mode().attention_kernels = std::move(kernels);
    }
    ~AttentionScope() { executor_.mode().attention_kernels = std::move(previous_); }
    AttentionScope(const AttentionScope&) = delete;
    AttentionScope& operator=(const AttentionScope&) = delete;
private:
    Executor& executor_;
    std::optional<AttentionKernels> previous_;
};

} // namespace tyloi::cuda
