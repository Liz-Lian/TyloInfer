#pragma once

#include "ops/vision/vision.h"

#include <cstddef>

namespace mfq::cuda::vision {

template <class Visitor>
void visit_linear(QuantLinear& weight, Visitor visit) {
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
             &weight.mxfp4.weight.values,
             &weight.mxfp4.weight.scales,
             &weight.mxfp8.weight.values,
             &weight.mxfp8.weight.scales,
             &weight.mxfp4_sq.weight.blob,
             &weight.mxfp4_sq.weight.row_q,
             &weight.mxfp4_sq.weight.row_symbol_byte_offsets,
             &weight.mxfp4_sq.weight.row_auxiliary,
             &weight.fp8_sq.weight.blob,
             &weight.fp8_sq.weight.row_q,
             &weight.fp8_sq.weight.row_symbol_byte_offsets,
         }) {
        if (tensor->defined()) visit(*tensor);
    }
    n.workspaces.clear();
    v.workspaces.clear();
}

template <class Visitor>
void visit_tensors(Affine& weight, Visitor visit) {
    visit_linear(weight.linear, visit);
    if (weight.bias.defined()) visit(weight.bias);
}

template <class Visitor>
void visit_tensors(LayerNorm& norm, Visitor visit) {
    visit(norm.weight);
    visit(norm.bias);
}

template <class Visitor>
void visit_tensors(Layer& layer, Visitor visit) {
    visit_tensors(layer.norm1, visit);
    visit_tensors(layer.norm2, visit);
    visit_tensors(layer.attention.qkv, visit);
    visit_tensors(layer.attention.output, visit);
    visit_tensors(layer.mlp_up, visit);
    visit_tensors(layer.mlp_down, visit);
}

template <class Weights, class Visitor>
void visit_tensors(Weights& weights, Visitor visit) {
    weights.visit_tensors(visit);
}

template <class Weights>
Weights to_cuda(Weights weights) {
    visit_tensors(weights,
        [](Tensor& tensor) { tensor = tensor.to(mfq_tensor_backend::kCUDA); });
    return weights;
}

template <class Weights>
std::size_t weight_bytes(Weights weights) {
    std::size_t bytes = 0;
    visit_tensors(weights,
        [&](Tensor& tensor) { bytes += tensor.nbytes(); });
    return bytes;
}

} // namespace mfq::cuda::vision
