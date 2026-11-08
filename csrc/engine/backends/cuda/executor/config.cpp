#include "config.h"

#include "engine/backends/cuda/native/collectives.h"
#include "engine/weights/cache.h"

#include <set>

namespace tyloi::cuda {

std::vector<tyloi::Device> cuda_devices(const CudaLoadConfig& config) {
    std::set<int> devices(config.layer_placement.devices.begin(), config.layer_placement.devices.end());
    devices.insert(config.tensor_parallel.devices.begin(), config.tensor_parallel.devices.end());
    devices.insert(config.expert_parallel.devices.begin(), config.expert_parallel.devices.end());
    devices.insert(config.layer_placement.primary_device());
    std::vector<tyloi::Device> result;
    result.reserve(devices.size());
    for (const int device : devices) result.emplace_back(tyloi::kCUDA, device);
    return result;
}

bool model_parallel_enabled(const CudaLoadConfig& config) {
    return config.tensor_parallel.enabled() || config.expert_parallel.enabled();
}

const ParallelConfig& model_parallel_config(const CudaLoadConfig& config) {
    return config.tensor_parallel.enabled() ? config.tensor_parallel : config.expert_parallel;
}

const ParallelConfig& moe_parallel_config(const CudaLoadConfig& config) {
    return config.expert_parallel.enabled() ? config.expert_parallel : config.tensor_parallel;
}

int model_parallel_primary_device(const CudaLoadConfig& config) {
    if (!config.tensor_parallel.devices.empty()) return config.tensor_parallel.primary_device();
    if (!config.expert_parallel.devices.empty()) return config.expert_parallel.primary_device();
    return 0;
}

bool model_parallel_cuda_graph_enabled(const CudaLoadConfig& config,
        const Executor& executor, const tyloi::weights::Cache& weights) {
    if (!weights.graph_safe(tyloi::weights::Cache::Graph::model_parallel,
            moe_parallel_config(config).enabled())) return false;
    if (!config.tensor_parallel.enabled() && !config.expert_parallel.enabled()) return true;
    const bool enabled = config.expert_parallel.enabled() && !config.tensor_parallel.enabled()
        ? config.expert_parallel_cuda_graph : config.tensor_parallel_cuda_graph;
    return executor.collectives().enabled() && enabled;
}

} // namespace tyloi::cuda
