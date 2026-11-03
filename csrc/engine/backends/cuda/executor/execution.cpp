#include "engine/backends/cuda/executor/execution.h"

#include "engine/backends/cuda/executor/tq.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/state/kv_offload.h"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <unordered_set>

CudaExecutionContext::CudaExecutionContext(
        mfq::cuda::RoutedCache& borrowed_weights,
        std::shared_ptr<mfq::cuda::KvMemoryManager>& borrowed_kv,
        std::shared_ptr<mfq::cuda::KvOffloadStore>& borrowed_offload)
    : weights(borrowed_weights),
      kv(borrowed_kv), offload(borrowed_offload) {
    auto [execution_config, options] = load_cuda_execution_config();
    config = std::move(execution_config);
    executor_ = std::make_unique<mfq::cuda::Executor>(
        std::move(options),
        model_parallel_primary_device(*this));
}

mfq::cuda::Projector CudaExecutionContext::projector() {
    return {executor().profiler(), executor().mmq(), *executor().tq(), executor().branches(), executor().collectives(), executor().options().projection,
            model_parallel_primary_device(*this), decode_graph_serial_branches,
            decode_graph_tp_projection_major};
}

std::unique_ptr<mfq::cuda::Executor> CudaExecutionContext::make_executor() {
    return executor_->fork(model_parallel_primary_device(*this));
}

mfq::cuda::Executor::Mode CudaExecutionContext::forward_mode() const {
    return {{force_moe_materialized_swiglu, force_moe_unfused_reduce,
             decode_graph_serial_branches, continuous_batch_cache_serial,
             continuous_batch_ffn_tile, continuous_batch_ffn_tiles,
             executor().options().moe_ffn.reuse_input, decode_graph_tp_projection_major},
            {executor().options().projection.grouped &&
                 config.tensor_parallel_shared_linear_attention_input,
             config.linear_conv_prefill_fused},
            executor().mmq().kv_cache_capacity,
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
    executor().pipelines().clear();
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
    attention_scratch.clear();
#endif
    executor_.reset();
}

void CudaExecutionContext::drain() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    if (executor().contexts().empty() && !weights.moe &&
        !weights.tq && !offload &&
        executor().tq()->empty() && executor().pipelines().empty() && executor().branches().empty() &&
        executor().collectives().streams().empty()) return;
#endif
    CudaScope scope(layer_placement.primary_device(), executor().contexts());
    std::exception_ptr failure;
    const auto finish = [&](auto&& work) {
        try { work(); } catch (...) { if (!failure) failure = std::current_exception(); }
    };
    finish([&] { drain_moe_expert_cache(weights.moe); });
    finish([&] { executor().pipelines().drain(); });
    finish([&] { executor().collectives().drain(); });
    if (offload) finish([&] { offload->flush(); });
    // shortcut: retirement fences the whole device; use events when every consumer has a completion handle.
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    for (const auto& [device, context] : executor().contexts()) finish([&] {
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
    finish([&] { executor().tq()->drain(); });
    finish([&] { executor().branches().drain(); });
    if (failure) std::rethrow_exception(failure);
}

void CudaExecutionContext::reset() {
    drain();
    executor().pipelines().clear();
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
    attention_scratch.clear();
#endif
    executor().branches().clear();
    executor().profiler().reset();
    executor().collectives().reset();
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
    executor().routes().clear();
    executor().mmq().reset();
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
    return execution.executor().collectives().enabled() && enabled;
}
