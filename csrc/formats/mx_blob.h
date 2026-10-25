#pragma once

#include <cstdint>
#include <vector>

namespace mfq::mx {

struct Fp8 {
    std::int64_t out = 0;
    std::int64_t neuron_len = 0;
    std::vector<std::uint8_t> values;
    std::vector<std::uint8_t> scales;
    int block_rows = 128;
    int block_columns = 128;
};

struct Fp4 {
    std::int64_t out = 0;
    std::int64_t neuron_len = 0;
    std::vector<std::uint8_t> values;
    std::vector<std::uint8_t> scales;
};

Fp8 unpack8(const std::vector<std::uint8_t>& blob);
Fp4 unpack4(const std::vector<std::uint8_t>& blob);

} // namespace mfq::mx
