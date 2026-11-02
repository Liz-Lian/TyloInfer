#pragma once
#include "engine/weights/encoding/format_features.h"
#include "engine/backends/cuda/executor/config.h"
#include "engine/weights/routed_cache.h"
#include "engine/weights/placement.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/ffn.h"
#include "engine/backends/cuda/executor/moe_ffn_pipeline.h"
#include "engine/backends/cuda/native/scope.h"
namespace mfq::cuda { class KvOffloadStore; class KvMemoryManager; }

#include "backend.h"
#include "engine/backends/cuda/native/branches.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/native/profile.h"
#include "diagnostics/route_stats.h"
#include "engine/backends/cuda/ops/projection/mmq.h"
#include "engine/backends/cuda/ops/projection/parallel.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/backends/cuda/ops/attention/scratch.h"
#include "engine/ops.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

class TqExec;

struct CudaExecutionContext {
    CudaExecutionContext(
        mfq::cuda::ContextSet& contexts,
        mfq::cuda::RoutedCache& weights,
        std::shared_ptr<mfq::cuda::KvMemoryManager>& kv,
        std::shared_ptr<mfq::cuda::KvOffloadStore>& offload,
        std::shared_ptr<TqExec>& tq);
    ~CudaExecutionContext();
    CudaExecutionContext(const CudaExecutionContext&) = delete;
    CudaExecutionContext& operator=(const CudaExecutionContext&) = delete;
    mfq::cuda::ContextSet& contexts;
    bool yarn_context_extension=false;
    mfq::ops::YarnScaling yarn;

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
    mfq::cuda::RoutedCache& weights;
    std::shared_ptr<mfq::cuda::KvMemoryManager>& kv;
    std::shared_ptr<mfq::cuda::KvOffloadStore>& offload;
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

bool model_parallel_enabled(const CudaExecutionContext& execution);
const ParallelConfig& model_parallel_config(
    const CudaExecutionContext& execution);
const ParallelConfig& moe_parallel_config(
    const CudaExecutionContext& execution);
int model_parallel_primary_device(const CudaExecutionContext& execution);
bool model_parallel_cuda_graph_enabled(const CudaExecutionContext& execution);
