#include "executor.h"

#include "engine/backends/cuda/executor/moe_ffn_pipeline.h"
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
#include "engine/backends/cuda/ops/norm/rms.h"
#include "engine/backends/cuda/ops/projection/mmq.h"
#include "diagnostics/route_stats.h"

#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tyloi::cuda {


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

std::vector<DenseTensor> Executor::storage() const {
    std::vector<DenseTensor> result;
    const auto append = [&](const DenseTensor& tensor) { if (tensor.defined()) result.push_back(tensor); };
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
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
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

namespace {

void record(Trace* trace, std::string_view name, const DenseTensor& value) {
    if (trace && !name.empty()) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        trace->emplace_back(std::string(name), std::make_shared<DenseTensor>(value.to(::tyloi::kFloat32).contiguous().clone()));
#else
        trace->emplace_back(std::string(name), value.to(::tyloi::kFloat32).contiguous().clone());
#endif
    }
}

std::vector<std::vector<size_t>> releases(std::span<const Task> tasks, const Inputs& inputs) {
    std::vector<bool> ready;
    for (const auto& value : inputs.values) ready.push_back(value.defined());
    const auto read = [&](size_t slot) {
        TYLOINFER_RUNTIME_CHECK(slot < ready.size() && ready[slot], "task input is unavailable");
    };
    const auto write = [&](size_t slot) {
        TYLOINFER_RUNTIME_CHECK(slot < ready.size(), "task output slot is out of range");
        ready[slot] = true;
    };
    for (const auto& task : tasks) {
        std::visit([&](const auto& op) {
            using Op = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<Op, task::Replay>) {
                TYLOINFER_RUNTIME_CHECK(op.state < inputs.linear.size() &&
                    inputs.linear[op.state].captured != nullptr,
                    "replay task inputs are incomplete");
            } else {
                if constexpr (std::is_same_v<Op, task::Concat>) {
                    read(op.first);
                    read(op.second);
                } else {
                    read(op.input);
                }
                if constexpr (std::is_same_v<Op, task::Dense>) read(op.residual);
                if constexpr (std::is_same_v<Op, task::Attention>) {
                    TYLOINFER_RUNTIME_CHECK(inputs.rope && op.state < inputs.kv.size() &&
                        op.residual != op.output, "attention task inputs are incomplete");
                    write(op.residual);
                }
                if constexpr (std::is_same_v<Op, task::Linear>) {
                    TYLOINFER_RUNTIME_CHECK(op.state < inputs.linear.size() &&
                        op.residual != op.output, "linear task inputs are incomplete");
                    write(op.residual);
                }
                if constexpr (std::is_same_v<Op, task::FullFfn> ||
                              std::is_same_v<Op, task::LinearFfn>) read(op.residual);
                write(op.output);
            }
        }, task.op);
    }
    if (inputs.output) read(*inputs.output);
    std::vector<bool> needed(ready.size(), false);
    if (inputs.output) needed[*inputs.output] = true;
    std::vector<std::vector<size_t>> result(tasks.size());
    for (size_t index = tasks.size(); index-- > 0;) {
        const auto write = [&](size_t slot) {
            needed[slot] = false;
            result[index].push_back(slot);
        };
        const auto read = [&](size_t slot) {
            if (!needed[slot]) result[index].push_back(slot);
            needed[slot] = true;
        };
        std::visit([&](const auto& op) {
            using Op = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<Op, task::Replay>) {
                return;
            } else {
                write(op.output);
                if constexpr (std::is_same_v<Op, task::Attention> ||
                              std::is_same_v<Op, task::Linear>) write(op.residual);
                if constexpr (std::is_same_v<Op, task::Concat>) {
                    read(op.first);
                    read(op.second);
                } else {
                    read(op.input);
                }
                if constexpr (std::is_same_v<Op, task::Dense> ||
                              std::is_same_v<Op, task::FullFfn> ||
                              std::is_same_v<Op, task::LinearFfn>) read(op.residual);
            }
        }, tasks[index].op);
    }
    return result;
}

} // namespace

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
          env_flag("TYLOINFER_MOE_ROUTE_OUTPUT_ENERGY", false))), routes_(*owned_routes_),
      owned_pipelines_(std::make_unique<Pipelines>(profiler_, mmq_, options_.projection.moe,
                                                   options_.moe_ffn, routes_)),
      pipelines_(*owned_pipelines_), primary_(primary) {
    profiler_.filter = options_.profile_filter;
    mode_.ffn.reuse_input = options_.moe_ffn.reuse_input;
    mode_.linear.shared_input = options_.projection.grouped && options_.tensor_parallel_shared_linear_attention_input;
    mode_.linear.fused_conv = options_.linear_conv_prefill_fused;
    mode_.transposed = options_.gdn_transposed_state;
    mode_.gemma4_fused_norms = options_.gemma4_fused_norms;
    mode_.groupwise_output = options_.dsv4_groupwise_output_a;
}

Executor::Executor(ContextSet& contexts, CudaProfiler& profiler, KlMmqState& mmq, const std::shared_ptr<TqExec>& tq,
        Branches& branches, Pipelines& pipelines, Collectives& collectives, RouteStats& routes,
        const Options& options, Mode& mode, int primary)
    : contexts_(contexts), options_(options), mode_(mode), profiler_(profiler), mmq_(mmq), tq_(tq), branches_(branches),
      collectives_(collectives), routes_(routes), pipelines_(pipelines), primary_(primary) {}

Executor::~Executor() = default;

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
void Executor::drain() {
    std::exception_ptr failure;
    try { join(); } catch (...) { failure = std::current_exception(); }
    try { wait(); } catch (...) { if (!failure) failure = std::current_exception(); }
    if (failure) std::rethrow_exception(failure);
}
#endif

void Executor::join() {
    std::exception_ptr failure;
    const auto finish = [&](auto&& work) {
        try { work(); } catch (...) { if (!failure) failure = std::current_exception(); }
    };
    finish([&] { pipelines_.drain(); });
    finish([&] { collectives_.drain(); });
    if (failure) std::rethrow_exception(failure);
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
void Executor::wait() {
#else
void Executor::wait(const std::unordered_set<int>& devices) {
#endif
    std::exception_ptr failure;
    const auto finish = [&](auto&& work) {
        try { work(); } catch (...) { if (!failure) failure = std::current_exception(); }
    };
    // shortcut: retirement fences the whole device; use events when every consumer has a completion handle.
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
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
    return std::unique_ptr<Executor>(new Executor(contexts_, profiler_, mmq_, tq_, branches_,
        pipelines_, collectives_, routes_, options_, mode_, primary));
}

Executor::Mode Executor::forward_mode(int primary) const {
    auto snapshot = mode_;
    snapshot.kv_capacity = mmq_.kv_cache_capacity;
    snapshot.primary = primary;
    return snapshot;
}

Projector Executor::projector(const Ffn::Mode& mode) {
    return projector(mode, primary_);
}

Projector Executor::projector(const Ffn::Mode& mode, int primary) {
    return {profiler_, mmq_, *tq_, branches_, collectives_, mfe_scratch_, moe_scratch_, options_.projection,
            primary, mode.serial, mode.projection_major};
}

AttentionScratch& Executor::attention_scratch(std::size_t index) {
    if (scratch_.size() <= index) scratch_.resize(index + 1);
    return scratch_[index];
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
AttentionScratch& Executor::attention_scratch(tyloi::Device device, AttentionGeometry geometry) {
    const auto stream = device.is_cuda()
        ? reinterpret_cast<std::uintptr_t>(current_stream(device.index).stream()) : 0;
    std::lock_guard lock(attention_scratch_mutex_);
    return attention_scratch_[{device.is_cuda() ? device.index : -1, stream}][geometry];
}
#endif

namespace {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
class ExecutorScope;
thread_local ExecutorScope* active_scope = nullptr;
#endif

class ExecutorScope final : public tyloi::Executor::Scope {
public:
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    ExecutorScope(Executor& executor, int primary, ContextSet& contexts, const Ffn::Mode& mode)
        : owner(executor), ffn(mode), native_(std::in_place, primary, contexts),
          no_grad_(std::in_place), previous_(active_scope) {
        active_scope = this;
    }
    ExecutorScope(Executor& executor, const Ffn::Mode& mode)
        : owner(executor), ffn(mode), previous_(active_scope) {
        if (!previous_ || &previous_->owner != &executor)
            throw std::logic_error("CUDA executor is not active");
        active_scope = this;
    }
    Executor& owner;
    Ffn::Mode ffn;
#else
    ExecutorScope(Executor& executor, int primary, ContextSet& contexts)
        : native_(primary, contexts) {
        (void)executor;
    }
#endif
    ~ExecutorScope() override {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        active_scope = previous_;
#endif
    }

private:
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    std::optional<CudaScope> native_;
    std::optional<::tyloi::cuda::NoGradGuard> no_grad_;
#else
    CudaScope native_;
    ::tyloi::cuda::NoGradGuard no_grad_;
#endif
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    ExecutorScope* previous_;
#endif
};
} // namespace

std::unique_ptr<tyloi::Executor::Scope> Executor::enter() {
    return enter(primary_);
}

std::unique_ptr<tyloi::Executor::Scope> Executor::enter(int device) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    return enter(device, active_scope && &active_scope->owner == this ? active_scope->ffn : mode_.ffn);
#else
    return std::make_unique<ExecutorScope>(*this, device, contexts_);
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
std::unique_ptr<tyloi::Executor::Scope> Executor::enter(int device, const Ffn::Mode& mode) {
    return std::make_unique<ExecutorScope>(*this, device, contexts_, mode);
}

std::unique_ptr<tyloi::Executor::Scope> Executor::enter(const Ffn::Mode& mode) {
    return std::make_unique<ExecutorScope>(*this, mode);
}

Executor& Executor::current() {
    if (!active_scope) throw std::runtime_error("no active CUDA executor scope");
    return active_scope->owner;
}

const Ffn::Mode& Executor::current_ffn() {
    if (!active_scope) throw std::runtime_error("no active CUDA executor scope");
    return active_scope->ffn;
}
#endif

DenseTensor Executor::run(std::span<const Task> tasks, Inputs inputs, const Mode& mode) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    auto scope = enter(mfq_current_cuda_device(), mode.ffn);
#endif
    const auto release = releases(tasks, inputs);
    auto options = options_;
    options.dense.mlp.nvq_fusion = options.projection.nvq_fusion;
    options.dense.bf16_norm = options.attention.fused_norm;
    const auto scratch_size = inputs.scratch_offset + tasks.size();
    if (scratch_.size() < scratch_size) scratch_.resize(scratch_size);
    const auto project = projector(mode.ffn);
    for (size_t index = 0; index < tasks.size(); ++index) {
        const auto& task = tasks[index];
        std::visit([&](const auto& op) {
            using Op = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<Op, task::Replay>) {
                const auto& view = inputs.linear[op.state];
                op.weights.replay(view.state, *view.captured, op.tokens);
            } else {
                auto* trace = task.scope >= 0 && task.scope == inputs.trace_scope ? inputs.trace : nullptr;
                auto& values = inputs.values;
                DenseTensor input, second;
                if constexpr (std::is_same_v<Op, task::Concat>) {
                    input = values[op.first];
                    second = values[op.second];
                } else {
                    input = values[op.input];
                }
                if constexpr (std::is_same_v<Op, task::Dense> ||
                              std::is_same_v<Op, task::FullFfn> ||
                              std::is_same_v<Op, task::LinearFfn>) second = values[op.residual];
                for (auto slot : release[index]) values[slot] = {};
                auto compute = [&]() -> DenseTensor {
                    if constexpr (std::is_same_v<Op, task::Norm>) {
                        return qwen_rms_norm(input.reshape({-1, input.size(-1)}).to(::tyloi::kFloat32),
                            op.weight, op.eps, op.offset).reshape(input.sizes());
                    } else if constexpr (std::is_same_v<Op, task::Embed>) {
                        return quant_embedding_lookup(op.weight, std::move(input));
                    } else if constexpr (std::is_same_v<Op, task::Softcap>) {
                        return op.cap > 0.0 ? ::tyloi::cuda::tanh(input / op.cap) * op.cap : std::move(input);
                    } else if constexpr (std::is_same_v<Op, task::Concat>) {
                        return ::tyloi::cuda::cat({input, second}, op.axis);
                    } else if constexpr (std::is_same_v<Op, task::Project>) {
                        return project.run(op.weight, std::move(input));
                    } else if constexpr (std::is_same_v<Op, task::Attention>) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                        auto output = op.weights.forward(std::move(input), inputs.attention,
                            inputs.kv[op.state], *inputs.rope, trace);
#else
                        auto output = op.weights.forward(project, profiler_, options.attention,
                            std::move(input), inputs.attention, inputs.kv[op.state],
                            scratch_[inputs.scratch_offset + index],
                            *inputs.rope, trace);
#endif
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                        if (op.residual_out) *op.residual_out = output[0];
                        values[op.residual] = std::move(output[0]);
                        return std::move(output[1]);
#else
                        if (op.residual_out) *op.residual_out = output[0];
                        values[op.residual] = std::move(output[0]);
                        return std::move(output[1]);
#endif
                    } else if constexpr (std::is_same_v<Op, task::Linear>) {
                        auto output = op.weights.forward(project, profiler_, inputs.linear_options,
                            std::move(input), inputs.linear[op.state].state,
                            inputs.linear[op.state].captured);
                        values[op.residual] = std::move(output[0]);
                        return std::move(output[1]);
                    } else if constexpr (std::is_same_v<Op, task::Dense>) {
                        return op.weights.forward(project, profiler_, mmq_, *tq_, branches_, collectives_,
                            options.projection, options.dense, mode.ffn.serial, std::move(second), std::move(input), trace);
                    } else if constexpr (std::is_same_v<Op, task::FullFfn>) {
                        return full_ffn(op.weights, op.norm, op.attention,
                            std::move(second), std::move(input), mode.ffn, trace);
                    } else if constexpr (std::is_same_v<Op, task::LinearFfn>) {
                        return linear_ffn(op.weights, op.norm, op.attention,
                            std::move(second), std::move(input), mode.ffn);
                    }
                };
                values[op.output] = task.profile.empty()
                    ? compute() : profiler_.measure(std::string(task.profile), compute);
                record(trace, task.trace, values[op.output]);
            }
        }, task.op);
    }
    return inputs.output ? std::move(inputs.values[*inputs.output]) : DenseTensor{};
}

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
