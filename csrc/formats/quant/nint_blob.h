#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace mfq {

inline constexpr std::uint8_t kNintAdaptiveStorageFlag = 0x80;

constexpr bool nint_has_adaptive_storage(std::uint8_t raw_bits) noexcept {
    return (raw_bits & kNintAdaptiveStorageFlag) != 0;
}

constexpr std::uint8_t nint_logical_bits(std::uint8_t raw_bits) noexcept {
    return raw_bits & ~kNintAdaptiveStorageFlag;
}

namespace nint {

struct Data {
    int format_version = 2;
    int bits = 0;
    int sub_bits = 0;
    int gs = 0;
    int axis = 0;
    int neuron_len = 0;
    std::vector<int64_t> shape;
    int out = 0;
    int ng = 0;
    std::vector<uint8_t> q_packed;
    std::vector<uint8_t> row_q_bits;
    std::vector<int64_t> row_q_bit_offsets;
    std::vector<uint8_t> row_sub_bits;
    std::vector<uint8_t> sub_scale;
    std::vector<uint8_t> sub_min;
    std::vector<uint16_t> neuron_scale_h;
    std::vector<uint16_t> neuron_min_h;
    double aggregate_bpw = 0.0;
    double distribution_entropy = 0.0;
};

Data unpack(const std::vector<uint8_t>& blob);
void describe(Data& data);

} // namespace nint

namespace nint8 {

struct Data {
    int axis = 0;
    int neuron_len = 0;
    std::vector<int64_t> shape;
    int out = 0;
    int ng = 0;
    std::vector<uint8_t> q;
    std::vector<uint16_t> scale_h;
};

Data unpack(const std::vector<uint8_t>& blob);

} // namespace nint8

} // namespace mfq
