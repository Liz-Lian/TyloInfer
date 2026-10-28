#pragma once

namespace mfq::metal::detail {
inline constexpr const char* kMlxAffineDecode = R"METAL(
template <int Q, typename W>
inline uint tinfer_affine_code(W values, uint row, uint column, uint width) {
    const uint bit = column * Q;
    const uint index = row * (width / 32u * Q) + bit / 32u;
    const uint shift = bit & 31u;
    uint value = values[index] >> shift;
    if (shift + Q > 32u) value |= values[index + 1u] << (32u - shift);
    return value & ((1u << Q) - 1u);
}
template <int Q, typename W>
inline uint4 tinfer_affine_codes4(W values, uint row, uint column, uint width) {
    const uint bit = column * Q;
    const uint index = row * (width / 32u * Q) + bit / 32u;
    const uint shift = bit & 31u;
    uint window = values[index] >> shift;
    if (shift + 4u * Q > 32u) window |= values[index + 1u] << (32u - shift);
    return (uint4(window) >> uint4(0u, Q, 2u * Q, 3u * Q)) & uint4((1u << Q) - 1u);
}
template <int Q, int GS, typename W, typename S, typename B>
inline float tinfer_affine_value(W values, S scales, B biases, uint row, uint column, uint width) {
    const uint group = row * (width / GS) + column / GS;
    return fma(float(scales[group]), float(tinfer_affine_code<Q>(values, row, column, width)), float(biases[group]));
}
)METAL";
} // namespace mfq::metal::detail
