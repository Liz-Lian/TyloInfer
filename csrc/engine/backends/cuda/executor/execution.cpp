#include "engine/backends/cuda/executor/execution.h"
#include "engine/drain.h"

#include <exception>
#include <iostream>

CudaExecutionContext::CudaExecutionContext(
        mfq::cuda::Executor& executor,
        mfq::cuda::CudaLoadConfig execution_config,
        mfq::cuda::RoutedCache& borrowed_weights,
        std::shared_ptr<mfq::cuda::KvMemoryManager>& borrowed_kv,
        std::shared_ptr<mfq::cuda::KvOffloadStore>& borrowed_offload)
    : config(std::move(execution_config)), weights(borrowed_weights),
      kv(borrowed_kv), offload(borrowed_offload), executor_(executor) {}

CudaExecutionContext::~CudaExecutionContext() {
    try { mfq::cuda::drain_cuda(executor(), weights, offload.get(),
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
