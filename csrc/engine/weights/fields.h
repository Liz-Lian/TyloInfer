#pragma once

#include "engine/weights/bias.h"

#include <cstddef>
#include <vector>

namespace tyloi::cuda::fields {

using DenseTensor = ::tyloi::DenseTensor;

template <class Visitor>
void tensors(QuantLinear& weight, Visitor visit) {
    TYLOINFER_RUNTIME_CHECK(
        !weight.tensor_parallel(),
        "streamed vision weights require one CUDA device");
    auto& n = weight.nint;
    auto& v = weight.nvq;
    for (auto* tensor : {
             &weight.dense,
             &n.q_packed,
             &n.row_q_bits,
             &n.row_q_bit_offsets,
             &n.q8_zero_scale,
             &n.sub_scale,
             &n.sub_min,
             &n.neuron_scale,
             &n.neuron_min,
             &n.route_metadata,
             &v.indices_packed,
             &v.aux_packed,
             &v.sub_scale_packed,
             &v.neuron_scale,
             &v.codebook,
             &v.integer_codebook,
             &v.decode_records,
             &weight.mxfp4.values,
             &weight.mxfp4.scales,
             &weight.mxfp8.values,
             &weight.mxfp8.scales,
             &weight.mxfp4_sq.blob,
             &weight.mxfp4_sq.row_q,
             &weight.mxfp4_sq.row_symbol_byte_offsets,
             &weight.mxfp4_sq.row_auxiliary,
             &weight.fp8_sq.blob,
             &weight.fp8_sq.row_q,
             &weight.fp8_sq.row_symbol_byte_offsets,
         }) {
        if (tensor->defined()) visit(*tensor);
    }
}

template <class Visitor>
void tensors(BiasWeight& weight, Visitor visit) {
    tensors(weight.weight, visit);
    if (weight.bias.defined()) visit(weight.bias);
}

template <class Weights, class Visitor>
void tensors(Weights& weights, Visitor visit) {
    weights.visit_tensors(visit);
}

template <class Weights>
Weights to_cuda(Weights weights) {
    tensors(weights,
        [](DenseTensor& tensor) { tensor = tensor.to(::tyloi::kCUDA); });
    return weights;
}

template <class Weights>
std::size_t bytes(Weights weights) {
    std::vector<const DenseTensor*> buffers;
    tensors(weights, [&](DenseTensor& tensor) { buffers.push_back(&tensor); });
    return ::tyloi::backing_bytes(buffers);
}

} // namespace tyloi::cuda::fields
