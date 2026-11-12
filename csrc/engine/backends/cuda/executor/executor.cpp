#include "executor.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/working_set.h"
#endif

#include "engine/backends/cuda/executor/tq.h"
#include "engine/backends/cuda/native/branches.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/backends/cuda/native/scope.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"
#include "engine/backends/cuda/kernels/gr_prefill_blas.h"
#include "engine/backends/cuda/ops/projection/mmq.h"
#include "diagnostics/route_stats.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/state/kv_memory.h"
#include "engine/state/kv_offload.h"
#endif

#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda {

std::vector<tyloi::Executor::Memory> Executor::memory(std::span<const tyloi::Device> devices,
        tyloi::Executor::MemoryDetail detail) const {
    if (&current().contexts_ != &contexts_)
        throw std::runtime_error("CUDA executor scope belongs to another resource owner");
    std::vector<tyloi::Executor::Memory> result;
    result.reserve(devices.size());
    for (const auto device : devices) {
        if (!device.is_cuda() || device.index < 0)
            throw std::invalid_argument("CUDA executor requires CUDA devices with non-negative indices");
        MfqCudaGuard guard(device.index);
        std::size_t available = 0, total = 0;
        TYLOINFER_CUDA_CHECK(cudaMemGetInfo(&available, &total));
        const auto stats = mfq_cuda_memory_stats(device.index);
        const auto context = default_context(device.index);
        tyloi::Executor::Memory snapshot{device, stats.allocated_bytes, stats.reserved_bytes, available, total,
            context->memory_stats().limit, context->resident_memory_usage(),
            context->available_memory(), context->workspace_available()};
        if (detail == tyloi::Executor::MemoryDetail::diagnostic) {
            snapshot.local = context->local_memory_usage();
            cudaMemPool_t pool = nullptr;
            std::uint64_t used = 0, reserved = 0;
            TYLOINFER_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, device.index));
            TYLOINFER_CUDA_CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used));
            TYLOINFER_CUDA_CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved));
            snapshot.default_pool_used = used;
            snapshot.default_pool_reserved = reserved;
        }
        result.push_back(snapshot);
    }
    return result;
}

void Executor::synchronize() {
    if (&current().contexts_ != &contexts_)
        throw std::runtime_error("CUDA executor scope belongs to another resource owner");
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
}

void Executor::trim(tyloi::Device device) {
    if (&current().contexts_ != &contexts_)
        throw std::runtime_error("CUDA executor scope belongs to another resource owner");
    if (!device.is_cuda() || device.index < 0)
        throw std::invalid_argument("CUDA executor requires CUDA devices with non-negative indices");
    MfqCudaGuard guard(device.index);
    default_context(device.index)->reclaim_idle_memory();
}

void Executor::limit(tyloi::Device device, std::size_t bytes) {
    if (&current().contexts_ != &contexts_)
        throw std::runtime_error("CUDA executor scope belongs to another resource owner");
    if (!device.is_cuda() || device.index < 0)
        throw std::invalid_argument("CUDA executor requires CUDA devices with non-negative indices");
    MfqCudaGuard guard(device.index);
    default_context(device.index)->set_memory_limit(bytes);
}

const MoeRoutePlan& MlaScratch::head_route(int64_t rows, int heads) {
    const int64_t key = rows * 4096 + heads;
    const auto build = [&] {
        auto options = ::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kInt32);
        auto ids = ::tyloi::cuda::remainder(::tyloi::cuda::arange(rows, options), heads).reshape({rows, 1}).contiguous();
        return build_moe_route_plan(ids, heads);
    };
    if (rows != heads) {
        if (transient_head_route_key != key) {
            transient_head_route = build();
            transient_head_route_key = key;
        }
        return transient_head_route;
    }
    auto found = head_routes.find(key);
    if (found != head_routes.end()) return found->second;
    return head_routes.emplace(key, build()).first->second;
}

void MlaScratch::ensure_meta() {
    constexpr int64_t kMetaFloats = 8 * 1024 * 1024;
    if (!attention_meta.defined() || attention_meta.numel() < kMetaFloats)
        attention_meta = ::tyloi::cuda::empty({kMetaFloats}, ::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kFloat32));
}

void MlaScratch::ensure_compressed() {
    ensure_meta();
    if (!hadamard_signs.defined())
        hadamard_signs = ::tyloi::cuda::ones({128}, ::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kInt8));
}

void MlaScratch::ensure_dense_workspace(int64_t B, int64_t M, int64_t logical_len) {
    const int64_t stride = (logical_len + 63) / 64 * 64;
    auto cuda = ::tyloi::TensorOptions().device(::tyloi::kCUDA);
    if (!dense_mask.defined() || dense_mask.size(0) < M || dense_mask.size(1) < stride)
        dense_mask = ::tyloi::cuda::empty({M, stride}, cuda.dtype(::tyloi::kFloat16));
    if (!kv_max.defined() || kv_max.numel() < B * M)
        kv_max = ::tyloi::cuda::empty({B * M}, cuda.dtype(::tyloi::kInt32));
    ensure_meta();
}

void MlaScratch::ensure_decode_workspace(int64_t B, int64_t planned_len) {
    const int64_t stride = (planned_len + 63) / 64 * 64;
    auto cuda = ::tyloi::TensorOptions().device(::tyloi::kCUDA);
    if (!decode_mask.defined() || decode_mask.size(0) != B || decode_mask.size(1) < stride)
        decode_mask = ::tyloi::cuda::empty({B, stride}, cuda.dtype(::tyloi::kFloat16));
    if (!kv_max.defined() || kv_max.numel() < B)
        kv_max = ::tyloi::cuda::empty({B}, cuda.dtype(::tyloi::kInt32));
    ensure_meta();
}

#ifdef TYLOINFER_NATIVE_CUDA
std::vector<::tyloi::Tensor> Executor::storage() const {
    std::vector<::tyloi::Tensor> result;
    const auto append = [&](const DenseTensor& tensor) {
        if (tensor.defined()) result.push_back(std::make_shared<DenseTensor>(tensor));
    };
#else
std::vector<DenseTensor> Executor::storage() const {
    std::vector<DenseTensor> result;
    const auto append = [&](const DenseTensor& tensor) { if (tensor.defined()) result.push_back(tensor); };
#endif
    const auto route = [&](const MoeRoutePlan& value) {
        append(value.ids); append(value.ids_dst); append(value.expert_bounds);
        append(value.tile_bounds); append(value.tile_experts);
        append(value.mma_tile_bounds); append(value.mma_tile_experts);
        append(value.wide_tile_bounds); append(value.wide_tile_experts);
        append(value.counts); append(value.cursors);
    };
    for (const auto& value : scratch_) {
        append(value.partial_o); append(value.partial_m); append(value.partial_l);
        append(value.mask); append(value.kv_max); append(value.meta);
    }
    {
        std::lock_guard lock(mfe_scratch_.mutex);
        for (const auto& [pool, streams] : mfe_scratch_.values) {
            (void)pool;
            for (const auto& [stream, buckets] : streams) {
                (void)stream;
                for (const auto& [key, value] : buckets) {
                    (void)key;
                    append(value.qx); append(value.xscale);
                }
            }
        }
    }
    {
        std::lock_guard lock(moe_scratch_.mutex);
        for (const auto& [owner, sources] : moe_scratch_.values) {
            (void)owner;
            for (const auto& [source, streams] : sources) {
                (void)source;
                for (const auto& [stream, scratch] : streams) {
                    (void)stream;
                    append(scratch->nint_expert_pool); append(scratch->nint_expert_local);
                    append(scratch->nvq_expert_pool); append(scratch->nvq_expert_local);
                    for (const auto& [rows, plan] : scratch->nint_input_plans) {
                        (void)rows;
                        append(plan.weight_ptrs); append(plan.quantize_descriptors);
                        append(plan.row_metadata_workspace);
                    }
                    for (const auto& [rows, plan] : scratch->nvq_active_plans) {
                        (void)rows; append(plan);
                    }
                    for (const auto& [key, workspace] : scratch->activation_workspaces) {
                        (void)key; append(workspace.qx); append(workspace.xscale);
                    }
                }
            }
        }
    }
#ifdef TYLOINFER_NATIVE_CUDA
    if (tq_) for (const auto& tensor : tq_->storage()) append(tensor);
    std::lock_guard lock(attention_scratch_mutex_);
    for (const auto& [key, geometries] : attention_scratch_) {
        (void)key;
        for (const auto& [geometry, value] : geometries) {
            (void)geometry;
            append(value.partial_o); append(value.partial_m); append(value.partial_l);
            append(value.mask); append(value.kv_max); append(value.meta);
        }
    }
#endif
    for (const auto& [device, value] : mla_scratch_) {
        (void)device;
        append(value.dense_mask); append(value.decode_mask); append(value.kv_max);
        append(value.attention_meta);
        for (const auto& [key, plan] : value.head_routes) { (void)key; route(plan); }
        route(value.transient_head_route);
        append(value.hadamard_signs);
    }
    return result;
}

Executor::Executor(Options options, int primary)
    : owned_contexts_(std::make_unique<ContextSet>()), contexts_(*owned_contexts_),
      owned_options_(std::make_unique<Options>(std::move(options))), options_(*owned_options_),
      owned_mode_(std::make_unique<Mode>()), mode_(*owned_mode_),
      owned_profiler_(std::make_unique<CudaProfiler>()),
      profiler_(*owned_profiler_), owned_mmq_(std::make_unique<KlMmqState>()),
      mmq_(*owned_mmq_), tq_(std::make_shared<TqExec>()),
      owned_branches_(std::make_unique<Branches>()), branches_(*owned_branches_),
      owned_collectives_(std::make_unique<Collectives>()), collectives_(*owned_collectives_),
      owned_routes_(std::make_unique<RouteStats>(env_value("TYLOINFER_MOE_ROUTE_STATS").value_or(""),
          env_flag("TYLOINFER_MOE_ROUTE_OUTPUT_ENERGY", false))), routes_(*owned_routes_), primary_(primary) {
    profiler_.filter = options_.profile_filter;
    mode_.linear.shared_input = options_.projection.grouped && options_.tensor_parallel_shared_linear_attention_input;
    mode_.linear.fused_conv = options_.linear_conv_prefill_fused;
    mode_.transposed = options_.gdn_transposed_state;
    mode_.gemma4_fused_norms = options_.gemma4_fused_norms;
    mode_.groupwise_output = options_.dsv4_groupwise_output_a;
}

Executor::Executor(ContextSet& contexts, CudaProfiler& profiler, KlMmqState& mmq, const std::shared_ptr<TqExec>& tq,
        Branches& branches, Collectives& collectives, RouteStats& routes,
        const Options& options, Mode& mode, int primary)
    : contexts_(contexts), options_(options), mode_(mode), profiler_(profiler), mmq_(mmq), tq_(tq), branches_(branches),
      collectives_(collectives), routes_(routes), primary_(primary) {}

Executor::~Executor() = default;

void Executor::reset() {
#ifndef TYLOINFER_NATIVE_CUDA
    attention_scratch_.clear();
#endif
    branches_.clear();
    profiler_.reset();
    collectives_.reset();
    mode_.ffn.serial = false;
    mode_.ffn.projection_major = false;
    mode_.ffn.batch_serial = false;
    routes_.clear();
    mmq_.reset();
    mode_.ffn.unfused = false;
    mode_.ffn.materialized = false;
    mode_.trace_layer = -1;
    mode_.trace = nullptr;
}

#ifdef TYLOINFER_NATIVE_CUDA
tyloi::engine::Metrics Executor::metrics() const {
    tyloi::engine::Metrics result;
    if (kv_) result = kv_->metrics();
    if (offload_) {
        const auto values = offload_->metrics();
        result.insert(result.end(), values.begin(), values.end());
    }
    return result;
}

void Executor::configure_kv(std::shared_ptr<KvMemoryManager> kv, std::shared_ptr<KvOffloadStore> offload) {
    kv_ = std::move(kv);
    offload_ = std::move(offload);
}

void Executor::bind_budget(tyloi::engine::Budget* budget) noexcept {
    if (kv_) kv_->bind_budget(budget);
    if (offload_) offload_->bind_budget(budget);
}

void Executor::drain() {
    std::exception_ptr failure;
    try { join(); } catch (...) { failure = std::current_exception(); }
    try { if (kv_) kv_->flush(); } catch (...) { if (!failure) failure = std::current_exception(); }
    try { if (offload_ && (!kv_ || offload_ != kv_->offload())) offload_->flush(); }
    catch (...) { if (!failure) failure = std::current_exception(); }
    try { wait(); } catch (...) { if (!failure) failure = std::current_exception(); }
    if (failure) std::rethrow_exception(failure);
}

void Executor::clear() {
    bind_budget(nullptr);
    weights_ = nullptr;
    kv_.reset();
    offload_.reset();
    if (env_frozen() && env_flag("TYLOINFER_CUDA_ENV_DUMP", false))
        std::cerr << "cuda knobs in use:\n" << env_dump();
    scratch_.clear();
    {
        std::lock_guard lock(attention_scratch_mutex_);
        attention_scratch_.clear();
    }
    mla_scratch_.clear();
    {
        std::lock_guard lock(mfe_scratch_.mutex);
        mfe_scratch_.values.clear();
    }
    {
        std::lock_guard lock(moe_scratch_.mutex);
        moe_scratch_.values.clear();
    }
    tq_->clear();
    branches_.clear();
}
#endif

void Executor::join() {
    std::exception_ptr failure;
    const auto finish = [&](auto&& work) {
        try { work(); } catch (...) { if (!failure) failure = std::current_exception(); }
    };
    finish([&] { collectives_.drain(); });
    if (failure) std::rethrow_exception(failure);
}

#ifdef TYLOINFER_NATIVE_CUDA
void Executor::wait() {
#else
void Executor::wait(const std::unordered_set<int>& devices) {
#endif
    std::exception_ptr failure;
    const auto finish = [&](auto&& work) {
        try { work(); } catch (...) { if (!failure) failure = std::current_exception(); }
    };
    // shortcut: retirement fences the whole device; use events when every consumer has a completion handle.
#ifdef TYLOINFER_NATIVE_CUDA
    for (const auto& [device, context] : contexts_) finish([&] {
#else
    for (int device : devices) finish([&] {
#endif
        MfqCudaGuard guard(device);
        TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());
    });
    finish([&] { tq_->drain(); });
    finish([&] { branches_.drain(); });
    if (failure) std::rethrow_exception(failure);
}

std::unique_ptr<Executor> Executor::fork() const {
    return fork(primary_);
}

std::unique_ptr<Executor> Executor::fork(int primary) const {
    auto result = std::unique_ptr<Executor>(new Executor(contexts_, profiler_, mmq_, tq_, branches_,
        collectives_, routes_, options_, mode_, primary));
#ifdef TYLOINFER_NATIVE_CUDA
    result->configure_kv(kv_, offload_);
    result->weights_ = weights_;
#endif
    return result;
}

Executor::Mode Executor::forward_mode(int primary) const {
    auto snapshot = mode_;
    snapshot.kv_capacity = mmq_.kv_cache_capacity;
    snapshot.primary = primary;
    return snapshot;
}

Projector Executor::projector(const FfnMode& mode) {
    return projector(mode, primary_);
}

Projector Executor::projector(const FfnMode& mode, int primary) {
    auto projection = options_.projection;
    if (mode.moe_prefill) projection.moe.prefill = *mode.moe_prefill;
    if (mode.moe_nvq_decode) projection.moe.nvq_decode = *mode.moe_nvq_decode;
    if (mode.moe_nint_decode) projection.moe.nint_decode = *mode.moe_nint_decode;
    return {profiler_, mmq_, *tq_, branches_, collectives_, mfe_scratch_, moe_scratch_, projection,
            primary, mode.serial, mode.projection_major, mode.independent};
}

AttentionScratch& Executor::attention_scratch(std::size_t index) {
    if (scratch_.size() <= index) scratch_.resize(index + 1);
    return scratch_[index];
}

#ifdef TYLOINFER_NATIVE_CUDA
AttentionScratch& Executor::attention_scratch(tyloi::Device device, AttentionGeometry geometry) {
    const auto stream = device.is_cuda()
        ? reinterpret_cast<std::uintptr_t>(current_stream(device.index).stream()) : 0;
    std::lock_guard lock(attention_scratch_mutex_);
    return attention_scratch_[{device.is_cuda() ? device.index : -1, stream}][geometry];
}
#endif

namespace {
#ifdef TYLOINFER_NATIVE_CUDA
class ExecutorProfile final : public tyloi::Executor::Profile {
public:
    ExecutorProfile(CudaProfiler& profiler, const char* name)
        : profiler_(profiler), previous_(profiler.enabled), name_(name) {
        try {
            profiler_.reset();
            profiler_.enabled = true;
            range_.emplace(profiler_, name_);
        } catch (...) {
            profiler_.enabled = previous_;
            throw;
        }
    }

    ~ExecutorProfile() override { profiler_.enabled = previous_; }

    void finish() override {
        range_->finish();
        range_.reset();
        profiler_.report(name_);
        profiler_.enabled = previous_;
    }

private:
    CudaProfiler& profiler_;
    bool previous_;
    std::string name_;
    std::optional<CudaProfiler::Range> range_;
};
#endif

class ExecutorScope;
thread_local ExecutorScope* active_scope = nullptr;

class ExecutorScope final : public tyloi::Executor::Scope {
public:
    Executor& owner;
#ifdef TYLOINFER_NATIVE_CUDA
    ExecutorScope(Executor& executor, int primary, ContextSet& contexts, const FfnMode& mode)
        : owner(executor), ffn(mode), native_(std::in_place, primary, contexts),
          no_grad_(std::in_place), previous_(active_scope) {
        if (previous_ && &previous_->owner == &executor)
            ffn.serial = ffn.serial || previous_->ffn.serial;
        active_scope = this;
    }
    ExecutorScope(Executor& executor, const FfnMode& mode)
        : owner(executor), ffn(mode), previous_(active_scope) {
        if (!previous_ || &previous_->owner != &executor)
            throw std::logic_error("CUDA executor is not active");
        // Capture serialization remains in force through nested policy scopes.
        ffn.serial = ffn.serial || previous_->ffn.serial;
        active_scope = this;
    }
    FfnMode ffn;
#else
    ExecutorScope(Executor& executor, int primary, ContextSet& contexts)
        : owner(executor), native_(primary, contexts), previous_(active_scope) {
        active_scope = this;
    }
#endif
    ~ExecutorScope() override {
        active_scope = previous_;
    }

private:
#ifdef TYLOINFER_NATIVE_CUDA
    std::optional<CudaScope> native_;
    std::optional<::tyloi::cuda::NoGradGuard> no_grad_;
#else
    CudaScope native_;
    ::tyloi::cuda::NoGradGuard no_grad_;
#endif
    ExecutorScope* previous_;
};
} // namespace

std::unique_ptr<tyloi::Executor::Scope> Executor::enter() {
    return enter(primary_);
}

std::unique_ptr<tyloi::Executor::Scope> Executor::enter(int device) {
#ifdef TYLOINFER_NATIVE_CUDA
    return enter(device, active_scope && &active_scope->owner == this ? active_scope->ffn : mode_.ffn);
#else
    return std::make_unique<ExecutorScope>(*this, device, contexts_);
#endif
}

std::unique_ptr<tyloi::Executor::Timer> Executor::timer() {
    return std::make_unique<internal::PrefillCudaTimer>();
}

#ifdef TYLOINFER_NATIVE_CUDA
std::unique_ptr<tyloi::Executor::Work> Executor::work() { return make_working_set(*this); }

std::unique_ptr<tyloi::Executor::Profile> Executor::profile(const char* name) {
    return std::make_unique<ExecutorProfile>(profiler_, name);
}

std::unique_ptr<tyloi::Executor::Scope> Executor::enter(int device, const FfnMode& mode) {
    return std::make_unique<ExecutorScope>(*this, device, contexts_, mode);
}

std::unique_ptr<tyloi::Executor::Scope> Executor::enter(const FfnMode& mode) {
    return std::make_unique<ExecutorScope>(*this, mode);
}

const FfnMode& Executor::current_ffn() {
    if (!active_scope) throw std::runtime_error("no active CUDA executor scope");
    return active_scope->ffn;
}
#endif

Executor& Executor::current() {
    if (!active_scope) throw std::runtime_error("no active CUDA executor scope");
    return active_scope->owner;
}
#ifdef TYLOINFER_NATIVE_CUDA
Executor* Executor::active() noexcept { return active_scope ? &active_scope->owner : nullptr; }
#endif

void Executor::prepare(std::int64_t vocabulary) {
    const auto context=tyloi::cuda::default_context(mfq_current_cuda_device());
    const auto before=context->local_memory_usage();
    {
        const auto gpu=::tyloi::TensorOptions().device(::tyloi::kCUDA);
        auto random=::tyloi::cuda::full({1},.37,gpu.dtype(::tyloi::kFloat32));
        auto counts=::tyloi::cuda::zeros({vocabulary},gpu.dtype(::tyloi::kInt32));
        for(auto dtype:{::tyloi::kFloat16,::tyloi::kFloat32}) {
            auto logits=::tyloi::cuda::zeros({1,vocabulary},gpu.dtype(dtype));
            // Lazy CUDA modules and sampler workspaces must be charged before
            // expert arenas consume the remaining device budget. Exercise both
            // compact and fallback sampling without touching request RNG/state.
            for(int k:{20,64,65})if(k<=vocabulary)
                (void)sample_top_k_top_p_cuda(logits,random,.7,k,.8);
            (void)sample_softmax_cuda(logits,random,.7);
            (void)sample_apply_penalties_cuda(logits,counts,.2,.05,1.0);
            if(vocabulary>=64)(void)sample_top_k_compact_cuda(logits,64);
        }
#if TYLOINFER_ENABLE_MFQ_COMPAT
        prepare_gr_prefill_blas_runtime();
#endif
    }
    context->reclaim_idle_memory();
    std::cerr<<"runtime_sampling_prepare vocab="<<vocabulary<<" local_before="<<before
        <<" local_after="<<context->local_memory_usage()<<"\n";
}

} // namespace tyloi::cuda
