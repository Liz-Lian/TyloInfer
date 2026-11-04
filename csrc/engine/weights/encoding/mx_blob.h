#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace tyloi::mx {

std::vector<std::byte> prefix(
    std::string_view dtype,
    const std::vector<std::int64_t>& logical_shape,
    const std::vector<std::int64_t>& storage_shape,
    const std::vector<std::int64_t>& scale_shape,
    const std::string& name);

} // namespace tyloi::mx

namespace tyloi::mxfp8 {

struct Data {
    std::int64_t out = 0;
    std::int64_t neuron_len = 0;
    std::vector<std::uint8_t> values;
    std::vector<std::uint8_t> scales;
    int block_rows = 128;
    int block_columns = 128;
};

Data unpack(const std::vector<std::uint8_t>& blob);

} // namespace tyloi::mxfp8

namespace tyloi::mxfp4 {

struct Data {
    std::int64_t out = 0;
    std::int64_t neuron_len = 0;
    std::vector<std::uint8_t> values;
    std::vector<std::uint8_t> scales;
};

Data unpack(const std::vector<std::uint8_t>& blob);

} // namespace tyloi::mxfp4
