#include "env.h"
#include "engine/backends/cuda/executor/options.h"
#include "backends/cuda/engine/executor/execution.h"
#include "engine/backends/cuda/executor/tq.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/state/kv_offload.h"

#include "engine/backends/cuda/kernels/mfq_cuda_quant_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

bool enabled_unless_disabled(const char* name) {
    return !mfq::cuda::env_flag(name, false);
}

} // namespace

CudaExecutionConfig load_cuda_execution_config() {
    CudaExecutionConfig result;
    result.profile_filter = mfq::cuda::env_value("TYLOINFER_PROFILE_CUDA_FILTER").value_or("");
    result.report_cuda_memory = mfq::cuda::env_flag("TYLOINFER_REPORT_CUDA_MEMORY", false);

    const auto parallel_graph = mfq::cuda::env_value("TYLOINFER_MODEL_PARALLEL_CUDA_GRAPH");
    if (parallel_graph) {
        result.model_parallel_cuda_graph = mfq::cuda::env_flag(
            "TYLOINFER_MODEL_PARALLEL_CUDA_GRAPH", true);
        result.tensor_parallel_cuda_graph = result.model_parallel_cuda_graph;
        result.expert_parallel_cuda_graph = result.model_parallel_cuda_graph;
    } else {
        result.tensor_parallel_cuda_graph = mfq::cuda::env_flag(
            "TYLOINFER_TP_CUDA_GRAPH", true);
        result.expert_parallel_cuda_graph = mfq::cuda::env_flag(
            "TYLOINFER_EP_CUDA_GRAPH", true);
    }
    result.tasks.projection.grouped = mfq::cuda::env_flag(
        "TYLOINFER_TP_GROUPED_PROJECTIONS", true);
    result.tensor_parallel_shared_linear_attention_input = mfq::cuda::env_flag(
        "TYLOINFER_TP_SHARED_LINEAR_ATTENTION_INPUT", true);
    result.tensor_parallel_mirror_linear_attention_scalars = mfq::cuda::env_flag(
        "TYLOINFER_TP_MIRROR_LINEAR_ATTENTION_SCALARS", true);
    result.tasks.projection.reduce_to_primary = mfq::cuda::env_value(
        "TYLOINFER_MODEL_PARALLEL_REDUCE_TO_PRIMARY")
        ? mfq::cuda::env_flag("TYLOINFER_MODEL_PARALLEL_REDUCE_TO_PRIMARY", true)
        : mfq::cuda::env_flag("TYLOINFER_TP_REDUCE_TO_PRIMARY", true);
    result.tasks.projection.fp16_reduce = mfq::cuda::env_value(
        "TYLOINFER_MODEL_PARALLEL_FP16_REDUCE")
        ? mfq::cuda::env_flag("TYLOINFER_MODEL_PARALLEL_FP16_REDUCE", true)
        : mfq::cuda::env_flag("TYLOINFER_TP_FP16_REDUCE", true);
    result.tasks.projection.peer_first = mfq::cuda::env_value(
        "TYLOINFER_MODEL_PARALLEL_PEER_FIRST_LAUNCH")
        ? mfq::cuda::env_flag("TYLOINFER_MODEL_PARALLEL_PEER_FIRST_LAUNCH", true)
        : mfq::cuda::env_flag("TYLOINFER_TP_PEER_FIRST_LAUNCH", true);

    result.tasks.projection.parallel = enabled_unless_disabled(
        "TYLOINFER_DISABLE_DECODE_BRANCH_PARALLEL");
    result.weights.shard_dense = mfq::cuda::env_flag(
        "TYLOINFER_TP_SHARD_NATIVE_FLOAT", false);
    result.tasks.projection.nvq_fusion = mfq::cuda::env_value("TYLOINFER_DISABLE_NVQ_FUSION")
        ? enabled_unless_disabled("TYLOINFER_DISABLE_NVQ_FUSION")
        : enabled_unless_disabled("TYLOINFER_DISABLE_NIQ_FUSION");
    result.weights.quant.nvq2 = mfq::cuda::env_flag("TYLOINFER_NVQ2_EXEC", false) &&
        (mfq::cuda::env_value("TYLOINFER_DISABLE_NVQ2_EXEC")
            ? enabled_unless_disabled("TYLOINFER_DISABLE_NVQ2_EXEC")
            : enabled_unless_disabled("TYLOINFER_DISABLE_NIQ2_EXEC"));
    result.weights.quant.nvq_groups = mfq::cuda::env_flag(
        "TYLOINFER_NVQ_EXTENDED_GROUP_EXEC", false);
    result.tasks.projection.moe.heterogeneous = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MOE_NVQ_HETERO");
    result.tasks.projection.moe.nvq_decode =
        enabled_unless_disabled("TYLOINFER_DISABLE_MOE_NVQ_HETERO_DECODE") &&
        !mfq::cuda::env_flag("TYLOINFER_NVQ_MOE_EXACT_REDUCTION", false) &&
        !mfq::cuda::env_value("TYLOINFER_NVQ_MOE_ROWS_PER_BLOCK") &&
        !mfq::cuda::env_value("TYLOINFER_NVQ_MOE_WARPS") &&
        !mfq::cuda::env_value("TYLOINFER_NVQ_MOE_SHARE_GROUP_STATE");
    result.tasks.projection.moe.active_decode = mfq::cuda::env_flag("TYLOINFER_MOE_NVQ_ACTIVE_DECODE",false);
    result.tasks.projection.moe.prefill = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MOE_PREFILL_MMA");
    result.tasks.projection.moe.min_tokens = mfq::cuda::env_int(
        "TYLOINFER_MOE_PREFILL_MMA_MIN_TOKENS", 9, 9,
        std::numeric_limits<int>::max());
    result.ffn.small_heterogeneous = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MOE_SMALL_HETERO");
    result.ffn.delayed_readback = mfq::cuda::env_flag(
        "TYLOINFER_MOE_DELAYED_ROUTE_READBACK", true);
    result.ffn.bundle_prefetch = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MOE_PROJECTION_BUNDLE_PREFETCH");
    result.moe_ffn.reuse_input = enabled_unless_disabled(
        "TYLOINFER_DISABLE_SPLIT_MOE_ACTIVATION_REUSE");
    result.moe_ffn.resident_dispatch = mfq::cuda::env_flag(
        "TYLOINFER_MOE_GPU_RESIDENT_DISPATCH", false);
    result.moe_ffn.fused_activation = mfq::cuda::env_flag(
        "TYLOINFER_MOE_FFN_FUSED_ACTIVATION", true);
    result.moe_ffn.two_stage = mfq::cuda::env_flag("TYLOINFER_MOE_TWO_STAGE_FFN", false);
    result.ffn.swiglu_fusion = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MOE_SWIGLU_QUANT_FUSION");
    result.ffn.reduce_fusion = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MOE_REDUCE_GATE_FUSION");
    result.tasks.dense.mlp.gelu_fusion = enabled_unless_disabled(
        "TYLOINFER_DISABLE_FFN_GEGLU_FUSION");
    result.tasks.dense.mlp.silu_fusion = enabled_unless_disabled(
        "TYLOINFER_DISABLE_FFN_SWIGLU_FUSION");
    result.tasks.dense.important_parallel = enabled_unless_disabled(
        "TYLOINFER_DISABLE_IN_BRANCH_PARALLEL");
    result.gemma4_fused_norms = mfq::cuda::env_flag(
        "TYLOINFER_GEMMA4_FUSED_NORMS", true);
    result.dsv4_groupwise_output_a = mfq::cuda::env_flag(
        "TYLOINFER_DSV4_GROUPWISE_OUTPUT_A", true);
    result.weights.grouped = enabled_unless_disabled(
        "TYLOINFER_DIAGNOSTIC_DISABLE_NINT_GROUP");
    result.tasks.dense.fp32_residual = mfq::cuda::env_flag(
        "TYLOINFER_DIAGNOSTIC_FP32_RESIDUAL", false);
    result.tasks.dense.important_fp32 = mfq::cuda::env_flag(
        "TYLOINFER_DIAGNOSTIC_IN_F32_DOWN", false);
    result.tasks.attention.write_aten = mfq::cuda::env_flag(
        "TYLOINFER_KV_CACHE_WRITE_ATEN", false);
    result.gdn_transposed_state = mfq::cuda::env_flag(
        "TYLOINFER_GDN_TRANSPOSED_STATE", true);
    result.gdn_fused_output = mfq::cuda::env_flag("TYLOINFER_GDN_FUSED_OUTPUT",true);
    result.linear_conv_prefill_fused = mfq::cuda::env_flag(
        "TYLOINFER_LINEAR_CONV_PREFILL_FUSED", true);
    result.tasks.attention.fused_rope = mfq::cuda::env_flag(
        "TYLOINFER_MINICPM_FUSED_BF16_ROPE", true);
    result.tasks.attention.fused_qk_rope_kv = mfq::cuda::env_flag(
        "TYLOINFER_MINICPM_FUSED_QK_NORM_ROPE_KV", true);
    result.tasks.attention.fused_norm = mfq::cuda::env_flag(
        "TYLOINFER_MINICPM_FUSED_BF16_RMSNORM", true);
    result.tasks.attention.fused_rope_kv = mfq::cuda::env_flag(
        "TYLOINFER_MINICPM_FUSED_ROPE_KV", true);
    result.tasks.attention.kernels.bf16_flash128 = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MINICPM_BF16_FLASH128");
    result.tasks.attention.kernels.flash128_casts = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MINICPM_FLASH128_SPECIALIZED_CASTS");
    result.tasks.attention.kernels.bf16_gqa_decode = mfq::cuda::env_flag(
        "TYLOINFER_MINICPM_BF16_GQA_DECODE", true);
    result.tasks.dense.bf16_swiglu = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MINICPM_BF16_SWIGLU_FUSION");
    result.tasks.dense.bf16_residual = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MINICPM_BF16_RESIDUAL_ACC");
    result.tasks.attention.kernels.mma_prefill = mfq::cuda::env_flag("TYLOINFER_MMA_ATTENTION", true);
    result.tasks.attention.kernels.mma_decode = mfq::cuda::env_flag(
        "TYLOINFER_MMA_ATTENTION_DECODE", true);
    result.tasks.attention.kernels.decode_aten = mfq::cuda::env_flag(
        "TYLOINFER_ATTENTION_DECODE_ATEN", false);
    result.tasks.attention.kernels.split_k = mfq::cuda::env_flag(
        "TYLOINFER_ATTENTION_DECODE_SPLITK", true);
    result.deepseek_v41_engram_cache_rows = mfq::cuda::env_size(
        "TYLOINFER_DEEPSEEK_V41_ENGRAM_CACHE_ROWS", 16'384);
    result.moe_mapped_gather = mfq::cuda::env_flag("TYLOINFER_MOE_MAPPED_GATHER", false);
    result.moe_mapped_copy_blocks = mfq::cuda::env_int(
        "TYLOINFER_MOE_MAPPED_COPY_BLOCKS", 64, 4, 128);
    result.moe_ssd_io_workers = mfq::cuda::env_int(
        "TYLOINFER_MOE_SSD_IO_WORKERS", 8, 1, 64);
    result.moe_ssd_overlap = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MOE_SSD_OVERLAP");
    result.weights.ranges = enabled_unless_disabled(
        "TYLOINFER_DISABLE_MOE_SSD_RANGES");
    result.moe_pipeline = mfq::cuda::env_flag("TYLOINFER_MOE_PIPELINE",false);
    result.moe_hybrid_cpu = mfq::cuda::env_flag("TYLOINFER_MOE_HYBRID_CPU",false);
    result.weights.preload = mfq::cuda::env_flag("TYLOINFER_MOE_PRELOAD_ALL",false);
    result.moe_assert_resident = mfq::cuda::env_flag("TYLOINFER_MOE_ASSERT_RESIDENT",false);
    result.moe_ram_pcie = mfq::cuda::env_flag("TYLOINFER_MOE_RAM_PCIE",false);
    result.moe_direct_ram = mfq::cuda::env_flag("TYLOINFER_MOE_DIRECT_RAM",true);
    result.moe_residency_adapt = mfq::cuda::env_flag("TYLOINFER_MOE_RESIDENCY_ADAPT",true);
    result.moe_residency_warm = mfq::cuda::env_flag("TYLOINFER_MOE_RESIDENCY_WARM",true);
    result.moe_residency_projection_heat = mfq::cuda::env_flag("TYLOINFER_MOE_RESIDENCY_PROJECTION_HEAT",true);
    result.residual.fused_projections = mfq::cuda::env_flag("TYLOINFER_GR_FUSED_PROJECTIONS",true);
    result.residual.fused_activation = mfq::cuda::env_flag("TYLOINFER_GR_FUSED_PROJECTION_ACTIVATION",true);
    result.residual.fused_post_norm = mfq::cuda::env_flag("TYLOINFER_GR_FUSED_POST_NORM",true);
    result.gdn_fused_preparation = mfq::cuda::env_flag("TYLOINFER_GDN_FUSED_PREPARATION",true);
    result.gdn_fused_core = mfq::cuda::env_flag("TYLOINFER_GDN_FUSED_CORE",true);
    result.residual.prepared_dense = mfq::cuda::env_flag("TYLOINFER_GR_PREPARED_DENSE_PROJECTION",false);
    result.residual.native_input = mfq::cuda::env_flag("TYLOINFER_GR_NATIVE_PROJECTION_INPUT",true);
    result.residual.fixed_group = mfq::cuda::env_flag("TYLOINFER_GR_FIXED_GROUP_PROJECTION",true);
    result.residual.compact_mix = mfq::cuda::env_flag("TYLOINFER_GR_COMPACT_MIX",true);
    result.residual.two_stage = mfq::cuda::env_flag("TYLOINFER_GR_TWO_STAGE",true);
    result.residual.dense_injection = mfq::cuda::env_flag("TYLOINFER_GR_TWO_STAGE_DENSE_INJECTION",true);
    result.tasks.projection.moe.nint_decode = mfq::cuda::env_flag("TYLOINFER_MOE_NINT_HETEROGENEOUS_DECODE",true);
    if (const auto value = mfq::cuda::env_value("TYLOINFER_MOE_RAM_PCIE_FRACTION")) {
        std::size_t end = 0;
        const auto fraction = std::stod(*value,&end);
        if (end != value->size() || !std::isfinite(fraction) || fraction < 0 || fraction > 1)
            throw std::invalid_argument("TYLOINFER_MOE_RAM_PCIE_FRACTION must be in [0,1]");
        result.moe_ram_pcie_fraction = fraction;
    }
    result.weights.quant.heterogeneous = result.tasks.projection.moe.heterogeneous;
    result.weights.mapped_embedding = result.moe_pipeline && result.moe_ram_pcie;
    return result;
}

using mfq_tensor_backend::indexing::Slice;

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

mfq::cuda::FfnExec CudaExecutionContext::ffn() {
    return {projector(), profiler, kl_mmq, *tq, branches, collectives, routes,
            config.tasks.projection, config.tasks.attention, config.tasks.dense, config.ffn,
            {force_moe_materialized_swiglu, force_moe_unfused_reduce,
             decode_graph_serial_branches, continuous_batch_cache_serial,
             continuous_batch_ffn_tile, continuous_batch_ffn_tiles,
             config.moe_ffn.reuse_input}};
}

std::unique_ptr<mfq::cuda::Executor> CudaExecutionContext::make_executor() {
    return std::make_unique<mfq::cuda::Executor>(
        profiler, kl_mmq, *tq, branches, collectives, routes, config.tasks, config.ffn,
        model_parallel_primary_device(*this));
}

mfq::cuda::Executor::Mode CudaExecutionContext::forward_mode() const {
    return {{force_moe_materialized_swiglu, force_moe_unfused_reduce,
             decode_graph_serial_branches, continuous_batch_cache_serial,
             continuous_batch_ffn_tile, continuous_batch_ffn_tiles,
             config.moe_ffn.reuse_input},
            {config.tasks.projection.grouped &&
                 config.tensor_parallel_shared_linear_attention_input,
             config.linear_conv_prefill_fused},
            kl_mmq.kv_cache_capacity,
            config.gdn_transposed_state,
            gemma_stage_trace,
            gemma_trace_layer,
            layer_placement.primary_device(),
            decode_graph_tp_projection_major};
}

CudaExecutionContext::~CudaExecutionContext() {
    try { drain(); }
    catch (const std::exception& error) {
        std::cerr << "runtime_execution_cleanup error=" << error.what() << '\n';
    }
    // Release pipelines while their borrowed branch streams still exist.
    pipelines.clear();
    attention_scratch.clear();
}

void CudaExecutionContext::drain() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    if (contexts.empty() && !weights.moe &&
        !weights.tq && !offload &&
        (!tq || tq->empty()) && pipelines.empty() && branches.empty() &&
        attention_scratch.empty() && collectives.streams().empty()) return;
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
    attention_scratch.clear();
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

void trace_gemma_stage(
        CudaExecutionContext& execution,
        int layer,
        const char * name,
        const mfq_tensor_backend::Tensor & value) {
    if (execution.gemma_stage_trace != nullptr &&
            layer == execution.gemma_trace_layer) {
        execution.gemma_stage_trace->emplace_back(
            name, value.to(mfq_tensor_backend::kFloat32).contiguous().clone());
    }
}
