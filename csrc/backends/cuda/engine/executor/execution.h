#pragma once
#include "formats/format_features.h"
#include "backends/cuda/engine/weights/routed_cache.h"
#include "backends/cuda/engine/weights/load.h"
#include "engine/executor/run.h"
#include "engine/executor/ffn.h"
#include "engine/executor/moe_ffn_pipeline.h"
#include "ops/common/include/gated_residual.h"
namespace mfq::cuda { class KvOffloadStore; class KvMemoryManager; }

#include "backend.h"
#include "native/cuda/branches.h"
#include "native/cuda/collectives.h"
#include "native/cuda/profile.h"
#include "diagnostics/route_stats.h"
#include "ops/common/include/mmq.h"
#include "ops/common/include/parallel.h"
#include "ops/common/include/projection.h"
#include "ops/attention/scratch.h"
#include "engine/weights/tensor_parallel.h"
#include "engine/models/common/yarn.h"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef TYLOINFER_CUDA_CHECK
#define TYLOINFER_CUDA_CHECK(expr) do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__)); \
    } \
} while (0)
#endif

class MoeExpertCache;
class TqCache;
class TqExec;

// One load owner outlives its execution context and every model borrowing it.
struct CudaStores {
    CudaStores() = default;
    CudaStores(const CudaStores&) = delete;
    CudaStores& operator=(const CudaStores&) = delete;
    std::shared_ptr<mfq::cuda::KvMemoryManager> kv_memory;
    mfq::cuda::RoutedCache weights;
    std::shared_ptr<mfq::cuda::KvOffloadStore> qsa_kv_store;

    // Drain execution before retiring backing; leases may retain individual handles.
    void reset() noexcept {
        qsa_kv_store.reset();
        kv_memory.reset();
        weights.reset();
    }
};

struct CudaExecutionConfig {
    mfq::cuda::weight_loader::Loader::Options weights;
    mfq::cuda::Executor::Options tasks;
    MoeFfnOptions moe_ffn;
    std::string profile_filter;
    bool report_cuda_memory = false;
    bool model_parallel_cuda_graph = true;
    bool tensor_parallel_cuda_graph = true;
    bool expert_parallel_cuda_graph = true;
    bool tensor_parallel_shared_linear_attention_input = true;
    bool tensor_parallel_mirror_linear_attention_scalars = true;
    mfq::cuda::FfnExec::Options ffn;
    bool gemma4_fused_norms = true;
    bool dsv4_groupwise_output_a = true;
    mfq::cuda::GatedResidual::Options residual;
    bool gdn_transposed_state = true;
    bool gdn_fused_output = true;
    bool gdn_fused_preparation = true;
    bool gdn_fused_core = true;
    bool linear_conv_prefill_fused = true;
    std::size_t deepseek_v41_engram_cache_rows = 16'384;
    bool moe_mapped_gather = false;
    int moe_mapped_copy_blocks = 64;
    int moe_ssd_io_workers = 8;
    bool moe_ssd_overlap = true;
    bool moe_pipeline = false;
    bool moe_hybrid_cpu = false;
    bool moe_assert_resident = false;
    bool moe_ram_pcie = false;
    bool moe_direct_ram = true;
    // Diagnostics may hold primary residency constant across paired passes.
    bool moe_residency_adapt = true;
    bool moe_residency_warm = true;
    bool moe_residency_projection_heat = true;
    std::optional<double> moe_ram_pcie_fraction;
};

CudaExecutionConfig load_cuda_execution_config();

struct CudaExecutionContext {
    CudaExecutionContext(CudaStores& stores, std::shared_ptr<TqExec>& tq);
    ~CudaExecutionContext();
    CudaExecutionContext(const CudaExecutionContext&) = delete;
    CudaExecutionContext& operator=(const CudaExecutionContext&) = delete;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    mfq::cuda::ContextSet native_contexts;
#endif
    bool yarn_context_extension=false;
    mfq::models::YarnScaling yarn;

    CudaProfiler profiler;
    CudaExecutionConfig config;
    bool force_moe_unfused_reduce = false;
    bool force_moe_materialized_swiglu = false;
    KlMmqState kl_mmq;
    std::unordered_set<int> dsv4_cpu_offload_layers;
    int64_t dsv4_cpu_offload_host_bytes = 0;
    int n_gpu_layers = -1;
    int dense_cpu_layer_count = 0;
    bool drop_file_cache = false;
    bool decode_graph_serial_branches = false;
    bool decode_graph_tp_projection_major = false;
    bool continuous_batch_cache_serial = false;
    int continuous_batch_ffn_tile = 0;
    int continuous_batch_ffn_tiles = 1;
    int gemma_trace_layer = -1;
    mfq::cuda::Trace* gemma_stage_trace = nullptr;
    ParallelConfig tensor_parallel;
    ParallelConfig expert_parallel;
    mfq::cuda::Collectives collectives;
    LayerPlacementConfig layer_placement;
    CudaStores& stores;
    const std::shared_ptr<TqExec>& tq;
    mfq::cuda::Branches branches;
    // Layer addresses identify reusable workspace; only execution retires it after drain.
    std::unordered_map<const void*, mfq::cuda::AttentionScratch> attention_scratch;
    mfq::cuda::RouteStats routes;
    mfq::cuda::Pipelines pipelines;

    mfq::cuda::Projector projector();
    mfq::cuda::FfnExec ffn();
    std::unique_ptr<mfq::cuda::Executor> make_executor();
    mfq::cuda::Executor::Mode forward_mode() const;
    void drain();
    void reset();
};

void mfq_release_host_allocator_cache() noexcept;
struct CudaScope {
    MfqCudaGuard device;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    mfq::cuda::ContextGuard context;
#endif
    explicit CudaScope(CudaExecutionContext& execution)
        : device(execution.layer_placement.primary_device())
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        , context(execution.native_contexts)
#endif
    {}
};

bool model_parallel_enabled(const CudaExecutionContext& execution);
const ParallelConfig& model_parallel_config(
    const CudaExecutionContext& execution);
const ParallelConfig& moe_parallel_config(
    const CudaExecutionContext& execution);
int model_parallel_primary_device(const CudaExecutionContext& execution);
bool model_parallel_cuda_graph_enabled(const CudaExecutionContext& execution);
void trace_gemma_stage(
    CudaExecutionContext& execution,
    int layer,
    const char* name,
    const mfq_tensor_backend::Tensor& value);
namespace mfq::cuda::internal {
void prepare_sampling_runtime(int64_t vocabulary);
}
