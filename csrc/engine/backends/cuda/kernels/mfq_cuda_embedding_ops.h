#pragma once

#include "backend.h"

#include <cstdint>

::tyloi::Tensor embedding_lookup_cuda(
    ::tyloi::Tensor weight, ::tyloi::Tensor token_ids);
::tyloi::Tensor nint_embedding_cuda(
    ::tyloi::Tensor q_packed, ::tyloi::Tensor row_q_bits,
    ::tyloi::Tensor row_q_bit_offsets,
    ::tyloi::Tensor sub_scale, ::tyloi::Tensor sub_min,
    ::tyloi::Tensor neuron_scale, ::tyloi::Tensor neuron_min,
    ::tyloi::Tensor token_ids, int64_t neuron_len, int64_t gs);
::tyloi::Tensor nint8_zero_embedding_lookup_cuda(
    ::tyloi::Tensor q, ::tyloi::Tensor scale, ::tyloi::Tensor token_ids,
    int64_t neuron_len);
::tyloi::Tensor mxfp8_embedding_lookup_cuda(
    ::tyloi::Tensor values, ::tyloi::Tensor scales, ::tyloi::Tensor token_ids);
::tyloi::Tensor mxfp4_embedding_lookup_cuda(
    ::tyloi::Tensor values, ::tyloi::Tensor scales, ::tyloi::Tensor token_ids);
::tyloi::Tensor nvq_embedding_lookup_cuda(
    ::tyloi::Tensor indices, ::tyloi::Tensor aux, ::tyloi::Tensor sub_scale,
    ::tyloi::Tensor neuron_scale, ::tyloi::Tensor codebook, ::tyloi::Tensor token_ids,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode);
