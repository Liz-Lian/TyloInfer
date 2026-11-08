#pragma once

#include "engine/config.h"
#include "engine/backends/cuda/executor/config.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

namespace tyloi::engine { class Engine; }
namespace tyloi::weights { class Cache; }

namespace tyloi::cuda {

class KvMemoryManager;
class KvOffloadStore;

struct CudaLoadOptions {
    std::string model_path, config_path, tokenizer_model;
    std::string cpu_offload_layers_arg, moe_cache_profile_path;
    std::string tensor_parallel_arg, tensor_split_arg;
    std::string expert_parallel_arg, expert_split_arg;
    std::string layer_parallel_arg, layer_split_arg;
    double moe_gpu_cache_gb = 0.0;
    int cpu_threads = 0;
    int64_t context_size = 0;
    bool parallel_test_duplicates = false;
    bool yarn_context_extension = false;
    bool n_gpu_layers_set = false;
    bool cpu_threads_set = false;
    int n_gpu_layers = -1;
};

struct CudaEngineOptions : CudaLoadOptions {
    int continuous_batching = 0;
    // Zero selects Auto; the resolved runtime limit is always positive.
    int64_t prefill_chunk_size = 0;
};

std::unique_ptr<tyloi::engine::Engine> load_cuda_engine(CudaEngineOptions options);
void reload_cuda_engine(tyloi::engine::Engine& engine, CudaEngineOptions options);

} // namespace tyloi::cuda

namespace tyloi::cuda::internal {

tyloi::engine::Config resolve_cuda_config(const CudaEngineOptions& options);
void setup_cuda_load(
    const CudaLoadOptions& options,
    CudaLoadConfig& config, Executor& executor, tyloi::weights::Cache& weights,
    std::shared_ptr<KvMemoryManager>& kv, std::shared_ptr<KvOffloadStore>& offload);

} // namespace tyloi::cuda::internal

namespace tyloi::cuda {

void finalize_kv(KvMemoryManager* kv, tyloi::weights::Cache& weights,
    std::shared_ptr<KvOffloadStore>& offload);
std::chrono::steady_clock::time_point finish_load(
    KvMemoryManager* kv, tyloi::weights::Cache& weights, bool report_memory);

} // namespace tyloi::cuda
