#include "engine/backends/cuda/executor/execution.h"
#include "engine/drain.h"
#include "engine/weights/cache.h"

#include <exception>
#include <iostream>

CudaExecutionContext::CudaExecutionContext(
        tyloi::cuda::Executor& executor,
        tyloi::cuda::CudaLoadConfig execution_config,
        ::tyloi::weights::Cache& borrowed_weights,
        std::shared_ptr<tyloi::cuda::KvMemoryManager>& borrowed_kv,
        std::shared_ptr<tyloi::cuda::KvOffloadStore>& borrowed_offload)
    : config(std::move(execution_config)), weights(borrowed_weights),
      kv(borrowed_kv), offload(borrowed_offload), executor_(executor) {}

CudaExecutionContext::~CudaExecutionContext() {
    try { tyloi::cuda::drain_cuda(executor(), weights, offload.get(),
                                config.layer_placement, config.tensor_parallel, config.expert_parallel); }
    catch (const std::exception& error) {
        std::cerr << "runtime_execution_cleanup error=" << error.what() << '\n';
    }
    // Release pipelines while their borrowed branch streams still exist.
    executor().pipelines().clear();
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
    executor().attention_scratch().clear();
#endif
}
