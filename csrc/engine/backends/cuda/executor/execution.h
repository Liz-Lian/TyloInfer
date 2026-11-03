#pragma once
#include "engine/weights/encoding/format_features.h"
#include "engine/backends/cuda/executor/config.h"
#include "engine/weights/routed_cache.h"
#include "engine/weights/placement.h"
#include "engine/backends/cuda/executor/executor.h"
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

struct CudaExecutionContext {
    CudaExecutionContext(
        mfq::cuda::RoutedCache& weights,
        std::shared_ptr<mfq::cuda::KvMemoryManager>& kv,
        std::shared_ptr<mfq::cuda::KvOffloadStore>& offload);
    ~CudaExecutionContext();
    CudaExecutionContext(const CudaExecutionContext&) = delete;
    CudaExecutionContext& operator=(const CudaExecutionContext&) = delete;
    bool yarn_context_extension=false;
    mfq::ops::YarnScaling yarn;

    CudaExecutionConfig config;
    bool force_moe_unfused_reduce = false;
    bool force_moe_materialized_swiglu = false;
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
    LayerPlacementConfig layer_placement;
    mfq::cuda::RoutedCache& weights;
    std::shared_ptr<mfq::cuda::KvMemoryManager>& kv;
    std::shared_ptr<mfq::cuda::KvOffloadStore>& offload;
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
    // Layer addresses identify reusable workspace; only execution retires it after drain.
    std::unordered_map<const void*, mfq::cuda::AttentionScratch> attention_scratch;
#endif
    mfq::cuda::Projector projector();
    mfq::cuda::Executor& executor() { return *executor_; }
    const mfq::cuda::Executor& executor() const { return *executor_; }
    std::unique_ptr<mfq::cuda::Executor> make_executor();
    mfq::cuda::Executor::Mode forward_mode() const;
    void drain();
    void reset();

private:
    std::unique_ptr<mfq::cuda::Executor> executor_;
};

bool model_parallel_enabled(const CudaExecutionContext& execution);
const ParallelConfig& model_parallel_config(
    const CudaExecutionContext& execution);
const ParallelConfig& moe_parallel_config(
    const CudaExecutionContext& execution);
int model_parallel_primary_device(const CudaExecutionContext& execution);
bool model_parallel_cuda_graph_enabled(const CudaExecutionContext& execution);
