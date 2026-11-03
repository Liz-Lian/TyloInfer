#pragma once

#include "engine/config.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/weights/loader.h"
#include "engine/weights/placement.h"
#include "engine/ops.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>

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

namespace mfq::cuda {

// Read once at owner construction; controls read on access stay in options.h.
struct CudaLoadConfig {
    weight_loader::Loader::Options weights;
    bool report_cuda_memory = false;
    bool model_parallel_cuda_graph = true;
    bool tensor_parallel_cuda_graph = true;
    bool expert_parallel_cuda_graph = true;
    bool tensor_parallel_mirror_linear_attention_scalars = true;
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
    bool yarn_context_extension = false;
    ops::YarnScaling yarn;
    std::unordered_set<int> dsv4_cpu_offload_layers;
    int64_t dsv4_cpu_offload_host_bytes = 0;
    int n_gpu_layers = -1;
    int dense_cpu_layer_count = 0;
    bool drop_file_cache = false;
    ParallelConfig tensor_parallel;
    ParallelConfig expert_parallel;
    LayerPlacementConfig layer_placement;
};

class KvMemoryManager;
class KvOffloadStore;
struct RoutedCache;

std::pair<CudaLoadConfig, Executor::Options> load_cuda_execution_config();
bool model_parallel_enabled(const CudaLoadConfig& config);
const ParallelConfig& model_parallel_config(const CudaLoadConfig& config);
const ParallelConfig& moe_parallel_config(const CudaLoadConfig& config);
int model_parallel_primary_device(const CudaLoadConfig& config);
bool model_parallel_cuda_graph_enabled(const CudaLoadConfig& config,
    const Executor& executor, const RoutedCache& weights);

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
void reload_cuda_engine(mfq::engine::Engine& engine, CudaEngineOptions options);

} // namespace mfq::cuda

namespace mfq::cuda::internal {

CudaRuntimeConfig resolve_cuda_runtime_config(const CudaEngineOptions& options);
void setup_cuda_load(
    const CudaLoadOptions& options,
    CudaLoadConfig& config, Executor& executor, RoutedCache& weights,
    std::shared_ptr<KvMemoryManager>& kv, std::shared_ptr<KvOffloadStore>& offload);

} // namespace mfq::cuda::internal

namespace mfq::cuda {

void finalize_kv(KvMemoryManager* kv, RoutedCache& weights,
    std::shared_ptr<KvOffloadStore>& offload);
std::chrono::steady_clock::time_point finish_load(
    KvMemoryManager* kv, RoutedCache& weights, std::size_t graph_reserve,
    bool report_memory);

} // namespace mfq::cuda
