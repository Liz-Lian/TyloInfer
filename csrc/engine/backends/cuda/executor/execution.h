#pragma once
#include "engine/weights/encoding/format_features.h"
#include "engine/load.h"
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
#include "engine/ops.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

struct CudaExecutionContext {
    CudaExecutionContext(
        mfq::cuda::Executor& executor,
        mfq::cuda::CudaLoadConfig config,
        mfq::cuda::RoutedCache& weights,
        std::shared_ptr<mfq::cuda::KvMemoryManager>& kv,
        std::shared_ptr<mfq::cuda::KvOffloadStore>& offload);
    ~CudaExecutionContext();
    CudaExecutionContext(const CudaExecutionContext&) = delete;
    CudaExecutionContext& operator=(const CudaExecutionContext&) = delete;
    mfq::cuda::CudaLoadConfig config;
    mfq::cuda::RoutedCache& weights;
    std::shared_ptr<mfq::cuda::KvMemoryManager>& kv;
    std::shared_ptr<mfq::cuda::KvOffloadStore>& offload;
    mfq::cuda::Executor& executor() { return executor_; }
    const mfq::cuda::Executor& executor() const { return executor_; }
private:
    mfq::cuda::Executor& executor_;
};
