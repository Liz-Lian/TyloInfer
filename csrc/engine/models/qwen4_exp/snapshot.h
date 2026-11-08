#pragma once

#include "engine/tensor.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace tyloi::models::qwen4_exp {

struct LayerSnapshot {
    std::shared_ptr<const ::tyloi::Tensor> keys, values, index;
    std::shared_ptr<const ::tyloi::Tensor> convolution, recurrent, ple_convolution;
    int64_t ple_batch = 0;
    std::vector<int64_t> ple_tokens;

    template <class Visit>
    void tensors(const Visit& visit) const {
        if (keys) visit(*keys);
        if (values) visit(*values);
        if (index) visit(*index);
        if (convolution) visit(*convolution);
        if (recurrent) visit(*recurrent);
        if (ple_convolution) visit(*ple_convolution);
    }
};

struct Snapshot {
    std::vector<LayerSnapshot> layers;
    std::shared_ptr<const ::tyloi::Tensor> positions;

    template <class Visit>
    void tensors(const Visit& visit) const {
        if (positions) visit(*positions);
        for (const auto& layer : layers) layer.tensors(visit);
    }
};

struct MtpSnapshot {
    std::vector<LayerSnapshot> layers;
    std::vector<std::shared_ptr<const ::tyloi::Tensor>> positions;
    std::vector<int64_t> lengths;

    template <class Visit>
    void tensors(const Visit& visit) const {
        for (const auto& layer : layers) layer.tensors(visit);
        for (const auto& position : positions) if (position) visit(*position);
    }
};

} // namespace tyloi::models::qwen4_exp
