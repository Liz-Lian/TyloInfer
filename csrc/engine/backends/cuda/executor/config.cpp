#include "config.h"

#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/weights/cache.h"

#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace {

void validate_parallel_config(
        const ParallelConfig& config,
        const char* devices_option,
        const char* split_option,
        int available_devices,
        bool allow_duplicate_devices) {
    if (config.devices.empty()) {
        if (!config.split.empty()) {
            throw std::runtime_error(
                std::string(split_option) + " requires " + devices_option);
        }
        return;
    }
    if (config.devices.size() < 2) {
        throw std::runtime_error(
            std::string(devices_option) +
            " requires at least two devices");
    }

    std::unordered_set<int> unique_devices;
    for (const int device : config.devices) {
        if (device < 0 || device >= available_devices) {
            throw std::runtime_error(
                std::string(devices_option) +
                " CUDA device is unavailable: " +
                std::to_string(device));
        }
        const bool inserted = unique_devices.insert(device).second;
        if (!allow_duplicate_devices && !inserted) {
            throw std::runtime_error(
                std::string(devices_option) +
                " CUDA devices must be unique");
        }
    }

    if (!config.split.empty()) {
        if (config.split.size() != config.devices.size()) {
            throw std::runtime_error(
                std::string(split_option) + " count must match " +
                devices_option + " devices");
        }
        for (const double weight : config.split) {
            if (!std::isfinite(weight) || weight <= 0.0) {
                throw std::runtime_error(
                    std::string(split_option) +
                    " values must be finite and positive");
            }
        }
    }
}

void print_parallel_config(
        const char* label,
        const ParallelConfig& config) {
    if (!config.enabled()) return;
    std::cerr << label << " devices=";
    for (std::size_t index = 0; index < config.devices.size(); ++index) {
        if (index) std::cerr << ',';
        std::cerr << config.devices[index];
    }
    if (!config.split.empty()) {
        std::cerr << " split=";
        for (std::size_t index = 0; index < config.split.size(); ++index) {
            if (index) std::cerr << ',';
            std::cerr << config.split[index];
        }
    }
    std::cerr << '\n';
}

} // namespace

namespace tyloi::cuda {

void configure_placement(CudaLoadConfig& config, Collectives& collectives) {
    collectives.reset();
    config.layer_placement.layer_devices.clear();

    int available = 0;
    TYLOINFER_CUDA_CHECK(cudaGetDeviceCount(&available));
    validate_parallel_config(
        config.tensor_parallel,
        "--tensor-parallel", "--tensor-split", available,
        config.tensor_parallel.allow_duplicate_devices);
    validate_parallel_config(
        config.expert_parallel,
        "--expert-parallel", "--expert-split", available,
        config.expert_parallel.allow_duplicate_devices);

    if (config.tensor_parallel.enabled() &&
            config.expert_parallel.enabled() &&
            config.tensor_parallel.devices !=
                config.expert_parallel.devices) {
        throw std::runtime_error(
            "combined tensor and expert parallelism requires the same "
            "ordered CUDA device group");
    }
    if (!config.tensor_parallel.enabled() &&
            !config.expert_parallel.enabled()) {
        TYLOINFER_CUDA_CHECK(cudaSetDevice(0));
    } else {
        const auto& group = config.tensor_parallel.enabled()
            ? config.tensor_parallel : config.expert_parallel;
        Collectives::enable_peer_access(group.devices);
        collectives.configure(
            group.devices, group.allow_duplicate_devices);
        TYLOINFER_CUDA_CHECK(cudaSetDevice(group.primary_device()));
        print_parallel_config(
            "tensor_parallel", config.tensor_parallel);
        print_parallel_config(
            "expert_parallel", config.expert_parallel);
        std::cerr << "model_parallel ranks=" << group.devices.size()
                  << " collective_backend="
                  << (collectives.enabled() ? "nccl" : "serial")
                  << '\n';
    }

    auto& placement = config.layer_placement;
    if (placement.devices.empty()) {
        if (!placement.split.empty()) {
            throw std::runtime_error(
                "--layer-split requires --layer-parallel");
        }
        placement.devices = {model_parallel_primary_device(config)};
        return;
    }
    if (config.tensor_parallel.enabled() ||
            config.expert_parallel.enabled()) {
        throw std::runtime_error(
            "--layer-parallel cannot be combined with tensor/expert parallelism");
    }

    const ParallelConfig layer_parallel{
        placement.devices, placement.split, false};
    validate_parallel_config(
        layer_parallel,
        "--layer-parallel", "--layer-split", available, false);
    TYLOINFER_CUDA_CHECK(cudaSetDevice(placement.primary_device()));
    print_parallel_config("layer_parallel", layer_parallel);
}

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
