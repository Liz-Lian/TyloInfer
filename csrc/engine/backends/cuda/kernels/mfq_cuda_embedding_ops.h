#pragma once

#include "backend.h"

#include <cstdint>

::tyloi::DenseTensor embedding_lookup_cuda(
    ::tyloi::DenseTensor weight, ::tyloi::DenseTensor token_ids);
::tyloi::DenseTensor nint_embedding_cuda(
    ::tyloi::DenseTensor q_packed, ::tyloi::DenseTensor row_q_bits,
    ::tyloi::DenseTensor row_q_bit_offsets,
    ::tyloi::DenseTensor sub_scale, ::tyloi::DenseTensor sub_min,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor neuron_min,
    ::tyloi::DenseTensor token_ids, int64_t neuron_len, int64_t gs);
::tyloi::DenseTensor nint8_zero_embedding_lookup_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor scale, ::tyloi::DenseTensor token_ids,
    int64_t neuron_len);
::tyloi::DenseTensor mxfp8_embedding_lookup_cuda(
    ::tyloi::DenseTensor values, ::tyloi::DenseTensor scales, ::tyloi::DenseTensor token_ids);
::tyloi::DenseTensor mxfp4_embedding_lookup_cuda(
    ::tyloi::DenseTensor values, ::tyloi::DenseTensor scales, ::tyloi::DenseTensor token_ids);
::tyloi::DenseTensor nvq_embedding_lookup_cuda(
    ::tyloi::DenseTensor indices, ::tyloi::DenseTensor aux, ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor codebook, ::tyloi::DenseTensor token_ids,
    int64_t neuron_len, int64_t gs, int64_t sub_bits, int64_t format, int64_t sign_mode);
