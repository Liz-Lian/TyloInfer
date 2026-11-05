#pragma once

#include "engine/weights/bias.h"
#include "engine/weights/sharded.h"

#include <cstddef>
#include <utility>
#include <vector>

namespace tyloi::cuda::fields {

using DenseTensor = ::tyloi::DenseTensor;

// Read-only enumeration of the buffers a record needs, for byte accounting.
template <class Visitor>
void tensors(const std::shared_ptr<const ::tyloi::Tensor>& weight, Visitor visit) {
    TYLOINFER_RUNTIME_CHECK(
        dynamic_cast<const ::tyloi::ShardedTensor*>(weight.get()) == nullptr,
        "streamed vision weights require one CUDA device");
    for (const auto* buffer : weight->buffers()) visit(*buffer);
}

template <class Visitor>
void tensors(const weights::BiasWeight& weight, Visitor visit) {
    tensors(weight.weight, visit);
    if (weight.bias && weight.bias->defined()) tensors(weight.bias, visit);
}

// Anything that enumerates its own tensors, such as a vision weight block.
template <class Weights, class Visitor,
    class = decltype(std::declval<const Weights&>().visit_tensors(std::declval<Visitor>()))>
void tensors(const Weights& weights, Visitor visit) {
    weights.visit_tensors(visit);
}

// Weight preparation: publish a copy of every tensor on `device`.
inline void to_device(std::shared_ptr<const ::tyloi::Tensor>& weight, ::tyloi::Device device) {
    weight = weight->relocated(device);
}

inline void to_device(weights::BiasWeight& weight, ::tyloi::Device device) {
    to_device(weight.weight, device);
    if (weight.bias && weight.bias->defined()) to_device(weight.bias, device);
}

template <class Weights>
void to_device(Weights& weights, ::tyloi::Device device) {
    weights.to_device(device);
}

template <class Weights>
Weights to_cuda(Weights weights) {
    to_device(weights, ::tyloi::Device(::tyloi::kCUDA, 0));
    return weights;
}

template <class Weights>
std::size_t bytes(const Weights& weights) {
    std::vector<const DenseTensor*> buffers;
    tensors(weights, [&](const DenseTensor& tensor) { buffers.push_back(&tensor); });
    return ::tyloi::backing_bytes(buffers);
}

} // namespace tyloi::cuda::fields
