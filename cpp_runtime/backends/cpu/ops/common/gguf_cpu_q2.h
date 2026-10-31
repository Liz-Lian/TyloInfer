#pragma once
#include <cstddef>
#include <cstdint>
#include "ggml-quants.h"
namespace mfq::cpu::gguf::simd {
// Four bit planes of two 64-value weight blocks share 128 activation bytes.
// The original Q8_0 packet precedes these temporary, per-token CPU views.
struct Q2Activation {
    std::int8_t planes[128];
    float scale[4],sum[4];
};
static_assert(sizeof(Q2Activation)==160);
inline std::size_t q2_activation_offset(int columns) {
    return (std::size_t(columns/32)*sizeof(block_q8_0)+31)&~std::size_t(31);
}
}
