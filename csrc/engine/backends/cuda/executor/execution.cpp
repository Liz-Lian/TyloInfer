#include "engine/backends/cuda/executor/execution.h"

#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/executor/tq.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/state/kv_offload.h"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

CudaExecutionContext::CudaExecutionContext(
        mfq::cuda::ContextSet& borrowed_contexts,
        mfq::cuda::RoutedCache& borrowed_weights,
        std::shared_ptr<mfq::cuda::KvMemoryManager>& borrowed_kv,
        std::shared_ptr<mfq::cuda::KvOffloadStore>& borrowed_offload,
        std::shared_ptr<TqExec>& borrowed_tq)
    : contexts(borrowed_contexts), config(load_cuda_execution_config()),
      weights(borrowed_weights),
      kv(borrowed_kv), offload(borrowed_offload), tq(borrowed_tq),
      routes(mfq::cuda::env_value("TYLOINFER_MOE_ROUTE_STATS").value_or(""),
             mfq::cuda::env_flag("TYLOINFER_MOE_ROUTE_OUTPUT_ENERGY", false)),
      pipelines(profiler, kl_mmq, config.tasks.projection.moe, config.moe_ffn, routes) {
    if (!tq) throw std::invalid_argument("CUDA execution context requires a TQ execution pool");
    profiler.filter = config.profile_filter;
}

mfq::cuda::Projector CudaExecutionContext::projector() {
    return {profiler, kl_mmq, *tq, branches, collectives, config.tasks.projection,
            model_parallel_primary_device(*this), decode_graph_serial_branches,
            decode_graph_tp_projection_major};
}

mfq::cuda::Executor CudaExecutionContext::executor() {
    return {contexts, profiler, kl_mmq, *tq, branches, collectives, routes, config.tasks, config.ffn,
            model_parallel_primary_device(*this)};
}

std::unique_ptr<mfq::cuda::Executor> CudaExecutionContext::make_executor() {
    return std::make_unique<mfq::cuda::Executor>(
        contexts, profiler, kl_mmq, *tq, branches, collectives, routes, config.tasks, config.ffn,
        model_parallel_primary_device(*this));
}

mfq::cuda::Executor::Mode CudaExecutionContext::forward_mode() const {
    return {{force_moe_materialized_swiglu, force_moe_unfused_reduce,
             decode_graph_serial_branches, continuous_batch_cache_serial,
             continuous_batch_ffn_tile, continuous_batch_ffn_tiles,
             config.moe_ffn.reuse_input, decode_graph_tp_projection_major},
            {config.tasks.projection.grouped &&
                 config.tensor_parallel_shared_linear_attention_input,
             config.linear_conv_prefill_fused},
            kl_mmq.kv_cache_capacity,
            config.gdn_transposed_state,
            gemma_stage_trace,
            gemma_trace_layer,
            layer_placement.primary_device(),
            config.gemma4_fused_norms,
            config.dsv4_groupwise_output_a};
}

CudaExecutionContext::~CudaExecutionContext() {
    try { drain(); }
    catch (const std::exception& error) {
        std::cerr << "runtime_execution_cleanup error=" << error.what() << '\n';
    }
    // Release pipelines while their borrowed branch streams still exist.
    pipelines.clear();
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
    attention_scratch.clear();
#endif
}

void CudaExecutionContext::drain() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    if (contexts.empty() && !weights.moe &&
        !weights.tq && !offload &&
        (!tq || tq->empty()) && pipelines.empty() && branches.empty() &&
        collectives.streams().empty()) return;
#endif
    CudaScope scope(layer_placement.primary_device(), contexts);
    std::exception_ptr failure;
    const auto finish = [&](auto&& work) {
        try { work(); } catch (...) { if (!failure) failure = std::current_exception(); }
    };
    finish([&] { drain_moe_expert_cache(weights.moe); });
    finish([&] { pipelines.drain(); });
    finish([&] { collectives.drain(); });
    if (offload) finish([&] { offload->flush(); });
    // shortcut: retirement fences the whole device; use events when every consumer has a completion handle.
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    for (const auto& [device, context] : contexts) finish([&] {
        MfqCudaGuard guard(device);
        TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());
    });
#else
    std::unordered_set<int> devices{layer_placement.primary_device()};
    for (const auto* group : {&tensor_parallel.devices, &expert_parallel.devices, &layer_placement.devices})
        devices.insert(group->begin(), group->end());
    for (int device : devices) finish([&] {
        MfqCudaGuard guard(device);
        TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());
    });
#endif
    if (tq) finish([&] { tq->drain(); });
    finish([&] { branches.drain(); });
    if (failure) std::rethrow_exception(failure);
}

void CudaExecutionContext::reset() {
    drain();
    pipelines.clear();
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
    attention_scratch.clear();
#endif
    branches.clear();
    profiler.reset();
    collectives.reset();
    tensor_parallel = {};
    expert_parallel = {};
    layer_placement = {};
    dsv4_cpu_offload_layers.clear();
    dsv4_cpu_offload_host_bytes = 0;
    n_gpu_layers = -1;
    dense_cpu_layer_count = 0;
    drop_file_cache = false;
    decode_graph_serial_branches = false;
    decode_graph_tp_projection_major = false;
    continuous_batch_cache_serial = false;
    routes.clear();
    kl_mmq.reset();
    force_moe_unfused_reduce = false;
    force_moe_materialized_swiglu = false;
    config.weights.min_slots = 8;
    gemma_trace_layer = -1;
    gemma_stage_trace = nullptr;
}

bool model_parallel_enabled(const CudaExecutionContext& execution) {
    return execution.tensor_parallel.enabled() ||
        execution.expert_parallel.enabled();
}

const ParallelConfig & model_parallel_config(
        const CudaExecutionContext& execution) {
    return execution.tensor_parallel.enabled()
        ? execution.tensor_parallel
        : execution.expert_parallel;
}

const ParallelConfig & moe_parallel_config(
        const CudaExecutionContext& execution) {
    return execution.expert_parallel.enabled()
        ? execution.expert_parallel
        : execution.tensor_parallel;
}

int model_parallel_primary_device(const CudaExecutionContext& execution) {
    if (!execution.tensor_parallel.devices.empty()) {
        return execution.tensor_parallel.primary_device();
    }
    if (!execution.expert_parallel.devices.empty()) {
        return execution.expert_parallel.primary_device();
    }
    return 0;
}

bool model_parallel_cuda_graph_enabled(
        const CudaExecutionContext& execution) {
    if(execution.weights.tq && moe_parallel_config(execution).enabled())return false;
    if (!execution.tensor_parallel.enabled() &&
            !execution.expert_parallel.enabled()) {
        return true;
    }
    const bool enabled = execution.expert_parallel.enabled() &&
            !execution.tensor_parallel.enabled()
        ? execution.config.expert_parallel_cuda_graph
        : execution.config.tensor_parallel_cuda_graph;
    return execution.collectives.enabled() && enabled;
}
