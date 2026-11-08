#pragma once

#include "engine/backends/cuda/executor/executor.h"
#include "engine/ops.h"
#include "engine/weights/loader.h"
#include "engine/weights/moe/options.h"
#include "engine/weights/placement.h"

#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tyloi::weights { class Cache; }

namespace tyloi::cuda {

// Read once at owner construction; controls read on access stay in options.h.
struct CudaLoadConfig {
    tyloi::weights::Loader::Options weights;
    bool report_cuda_memory = false;
    bool model_parallel_cuda_graph = true;
    bool tensor_parallel_cuda_graph = true;
    bool expert_parallel_cuda_graph = true;
    bool tensor_parallel_mirror_linear_attention_scalars = true;
    std::size_t deepseek_v41_engram_cache_rows = 16'384;
    tyloi::weights::CachePolicy cache;
    bool yarn_context_extension = false;
    tyloi::ops::YarnScaling yarn;
    std::unordered_set<int> dsv4_cpu_offload_layers;
    int64_t dsv4_cpu_offload_host_bytes = 0;
    int n_gpu_layers = -1;
    int dense_cpu_layer_count = 0;
    bool drop_file_cache = false;
    ParallelConfig tensor_parallel;
    ParallelConfig expert_parallel;
    LayerPlacementConfig layer_placement;
};

std::pair<CudaLoadConfig, Executor::Options> load_cuda_execution_config();
std::vector<tyloi::Device> cuda_devices(const CudaLoadConfig& config);
bool model_parallel_enabled(const CudaLoadConfig& config);
const ParallelConfig& model_parallel_config(const CudaLoadConfig& config);
const ParallelConfig& moe_parallel_config(const CudaLoadConfig& config);
int model_parallel_primary_device(const CudaLoadConfig& config);
bool model_parallel_cuda_graph_enabled(const CudaLoadConfig& config,
    const Executor& executor, const tyloi::weights::Cache& weights);

} // namespace tyloi::cuda
