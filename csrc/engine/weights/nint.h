#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

struct NintWeight {
    ::tyloi::Tensor q_packed;
    ::tyloi::Tensor row_q_bits;
    ::tyloi::Tensor row_q_bit_offsets;
    ::tyloi::Tensor q8_zero_scale;
    ::tyloi::Tensor sub_scale;
    ::tyloi::Tensor sub_min;
    ::tyloi::Tensor neuron_scale;
    ::tyloi::Tensor neuron_min;
    // Experimental routed row records; populated only by the comparison bench.
    ::tyloi::Tensor route_metadata;
    int64_t out = 0;
    int64_t ng = 0;
    int64_t gs = 0;
    int64_t bits = 0;
    int64_t neuron_len = 0;
    int64_t q_expert_stride = 0;
    int format_version = 2;
    double aggregate_bpw = 0.0;
    double distribution_entropy = 0.0;
    bool q8_zero = false;
    // Proven once from CPU row metadata; false for unclassified/grouped weights.
    bool aligned_q8 = false;
    // Proven from immutable CPU row metadata; zero means mixed/unknown.
    int uniform_qbits = 0;
    std::vector<int64_t> shape;
};
