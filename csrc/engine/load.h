#pragma once

#include "engine/config.h"
#include "engine/backends/cuda/executor/options.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace mfq::engine { class Engine; }

namespace mfq::cuda {

struct CudaContinuousBatchConfig : mfq::engine::ContinuousBatchConfig {
    bool greedy = true;
    bool cuda_graph = true;
    bool paged_kv = true;
    std::int32_t cuda_graph_minimum_tokens = 16;
};

struct CudaRuntimeConfig : mfq::engine::RuntimeConfig {
    bool prefill_chunk_auto = true;
    CudaDecodeGraphConfig decode_graph;
    CudaContinuousBatchConfig continuous_batch;
};

} // namespace mfq::cuda

struct CudaExecutionContext;

namespace mfq::cuda {

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

std::unique_ptr<mfq::engine::Engine> load_cuda_engine(CudaEngineOptions options);

} // namespace mfq::cuda

namespace mfq::cuda::internal {

CudaRuntimeConfig resolve_cuda_runtime_config(const CudaEngineOptions& options);
void setup_cuda_load(
    const CudaLoadOptions& options,
    CudaExecutionContext& execution);

} // namespace mfq::cuda::internal

namespace mfq::cuda {

class KvMemoryManager;
class KvOffloadStore;
struct RoutedCache;

void finalize_kv(KvMemoryManager* kv, RoutedCache& weights,
    std::shared_ptr<KvOffloadStore>& offload);
std::chrono::steady_clock::time_point finish_load(
    KvMemoryManager* kv, RoutedCache& weights, std::size_t graph_reserve,
    bool report_memory);

} // namespace mfq::cuda
