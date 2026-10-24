#pragma once

#include "backend.h"

#include <cstdint>

mfq_tensor_backend::Tensor embedding_lookup_cuda(
    mfq_tensor_backend::Tensor weight, mfq_tensor_backend::Tensor token_ids);
mfq_tensor_backend::Tensor nint_embedding_cuda(
    mfq_tensor_backend::Tensor q_packed, mfq_tensor_backend::Tensor row_q_bits,
    mfq_tensor_backend::Tensor row_q_bit_offsets,
    mfq_tensor_backend::Tensor sub_scale, mfq_tensor_backend::Tensor sub_min,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor neuron_min,
    mfq_tensor_backend::Tensor token_ids, int64_t neuron_len, int64_t gs);
mfq_tensor_backend::Tensor nint8_zero_embedding_lookup_cuda(
    mfq_tensor_backend::Tensor q, mfq_tensor_backend::Tensor scale, mfq_tensor_backend::Tensor token_ids,
    int64_t neuron_len);
mfq_tensor_backend::Tensor mxfp8_embedding_lookup_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales, mfq_tensor_backend::Tensor token_ids);
mfq_tensor_backend::Tensor mxfp4_embedding_lookup_cuda(
    mfq_tensor_backend::Tensor values, mfq_tensor_backend::Tensor scales, mfq_tensor_backend::Tensor token_ids);
mfq_tensor_backend::Tensor nvq_embedding_lookup_cuda(
    mfq_tensor_backend::Tensor indices, mfq_tensor_backend::Tensor aux, mfq_tensor_backend::Tensor sub_scale,
    mfq_tensor_backend::Tensor neuron_scale, mfq_tensor_backend::Tensor codebook, mfq_tensor_backend::Tensor token_ids,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode);
