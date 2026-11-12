#pragma once

#include "engine/weights/bias.h"
#include "engine/weights/sharded.h"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::weights::fields {

/** Enumerate owning tensor handles and reject unsupported sharded streaming. */
template <class Visitor>
void tensors(const ::tyloi::Tensor& weight, Visitor visit) {
    if (weight.representation<::tyloi::ShardedTensor>() != nullptr)
        throw std::runtime_error("streamed vision weights require one CUDA device");
    if (!weight || !weight.defined()) return;
    visit(weight);
}

template <class Visitor>
void tensors(const ::tyloi::weights::BiasWeight& weight, Visitor visit) {
    tensors(weight.weight, visit);
    if (weight.bias && weight.bias.defined()) tensors(weight.bias, visit);
}

// Anything that enumerates its own tensors, such as a vision weight block.
template <class Weights, class Visitor,
    class = decltype(std::declval<const Weights&>().visit_tensors(std::declval<Visitor>()))>
void tensors(const Weights& weights, Visitor visit) {
    weights.visit_tensors(visit);
}

// Weight preparation: publish a copy of every tensor on `device`.
inline void to_device(::tyloi::Tensor& weight, ::tyloi::Device device) {
    weight = weight.relocated(device);
}

inline void to_device(::tyloi::weights::BiasWeight& weight, ::tyloi::Device device) {
    to_device(weight.weight, device);
    if (weight.bias && weight.bias.defined()) to_device(weight.bias, device);
}

template <class Weights>
void to_device(Weights& weights, ::tyloi::Device device) {
    weights.to_device(device);
}

template <class Weights>
std::size_t bytes(const Weights& weights) {
    std::vector<const ::tyloi::DenseTensor*> buffers;
    tensors(weights, [&](const ::tyloi::Tensor& tensor) {
        const auto parts = tensor.buffers();
        buffers.insert(buffers.end(), parts.begin(), parts.end());
    });
    return ::tyloi::backing_bytes(buffers);
}

} // namespace tyloi::weights::fields
