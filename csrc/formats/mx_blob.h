#pragma once

#include <cstdint>
#include <vector>

namespace mfq::mxfp8 {

struct Data {
    std::int64_t out = 0;
    std::int64_t neuron_len = 0;
    std::vector<std::uint8_t> values;
    std::vector<std::uint8_t> scales;
    int block_rows = 128;
    int block_columns = 128;
};

Data unpack(const std::vector<std::uint8_t>& blob);

} // namespace mfq::mxfp8

namespace mfq::mxfp4 {

struct Data {
    std::int64_t out = 0;
    std::int64_t neuron_len = 0;
    std::vector<std::uint8_t> values;
    std::vector<std::uint8_t> scales;
};

Data unpack(const std::vector<std::uint8_t>& blob);

} // namespace mfq::mxfp4
