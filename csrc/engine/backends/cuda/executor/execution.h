#pragma once
#include "engine/weights/encoding/format_features.h"
#include "engine/load.h"
#include "engine/weights/placement.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/moe_ffn_pipeline.h"
#include "engine/backends/cuda/native/scope.h"
namespace tyloi::cuda { class KvOffloadStore; class KvMemoryManager; }
namespace tyloi::weights { class Cache; }

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
        tyloi::cuda::Executor& executor,
        tyloi::cuda::CudaLoadConfig config,
        ::tyloi::weights::Cache& weights,
        std::shared_ptr<tyloi::cuda::KvMemoryManager>& kv,
        std::shared_ptr<tyloi::cuda::KvOffloadStore>& offload);
    ~CudaExecutionContext();
    CudaExecutionContext(const CudaExecutionContext&) = delete;
    CudaExecutionContext& operator=(const CudaExecutionContext&) = delete;
    tyloi::cuda::CudaLoadConfig config;
    ::tyloi::weights::Cache& weights;
    std::shared_ptr<tyloi::cuda::KvMemoryManager>& kv;
    std::shared_ptr<tyloi::cuda::KvOffloadStore>& offload;
    tyloi::cuda::Executor& executor() { return executor_; }
    const tyloi::cuda::Executor& executor() const { return executor_; }
private:
    tyloi::cuda::Executor& executor_;
};
