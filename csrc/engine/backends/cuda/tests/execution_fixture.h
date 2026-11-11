#pragma once

// Minimal CudaExecutionContext fixture for kernel tests: no weights are
// loaded, only the executor/context machinery a Block forward needs.

#include "engine/backends/cuda/executor/execution.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/options.h"

#include <memory>
#include <utility>

struct TestExecution {
    std::unique_ptr<tyloi::cuda::Executor> executor;
    tyloi::cuda::CudaLoadConfig config;
    tyloi::cuda::RoutedCache weights;
    std::shared_ptr<tyloi::cuda::KvMemoryManager> kv;
    std::shared_ptr<tyloi::cuda::KvOffloadStore> offload;
    std::unique_ptr<CudaExecutionContext> execution;

    TestExecution() {
        auto [loaded, numeric] = tyloi::cuda::load_cuda_execution_config();
        config = std::move(loaded);
        executor = std::make_unique<tyloi::cuda::Executor>(std::move(numeric), 0);
        execution = std::make_unique<CudaExecutionContext>(
            *executor, config, weights, kv, offload);
    }

    operator CudaExecutionContext&() { return *execution; }
    CudaExecutionContext* operator->() { return execution.get(); }
};
