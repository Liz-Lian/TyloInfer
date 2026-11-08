#pragma once

#include "engine/backends/cuda/executor/execution.h"
#include "engine/weights/loader.h"

#include <iostream>
#include <memory>
#include <string>

namespace tyloi::cuda::detail {

template <class Model> struct ModelLoader {
    Model &model;
    const tyloi::weights::Loader& weights;
    auto embedding(const std::string& name) {
        return weights.embedding(name,
                              model.has_independent_output_weights(*this));
    }
    auto output(const std::string &name) {
        return weights.weight(name, TensorParallelAxis::Output);
    }
    auto tied_output(const std::shared_ptr<const ::tyloi::Tensor> &embedding, const std::string &name) {
        return weights.placement.tensor.enabled() ? output(name) : embedding;
    }
    bool has_weight(const std::string &name) { return weights.has(name); }
    void final_state() {
        model.adapter_load_final_state(weights);
    }
    void prepare_blocks() { model.prepare_blocks(weights); }
    auto block(int layer, const std::string &type) {
        auto &execution = *model.execution;
        const int device = execution.config.layer_placement.device_for_layer(layer);
        const bool cpu_offloaded = layer < execution.config.dense_cpu_layer_count;
        MfqCudaGuard layer_guard(device);
        std::cerr << "loading layer " << layer << ' ' << type << ' '
                  << (cpu_offloaded ? "CPU" : "CUDA") << std::endl;
        auto placed = weights.for_layer(device, cpu_offloaded);
        auto block = model.adapter_load_block(placed, layer, device, type);
        block->cuda_device = device;
        block->cpu_offloaded = cpu_offloaded;
        return block;
    }
};

} // namespace tyloi::cuda::detail
