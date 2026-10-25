#include "mx_blob.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace mfq {
namespace {

static std::uint64_t u64(const std::vector<std::uint8_t>& data, std::size_t& offset) {
    std::uint64_t result;
    std::memcpy(&result, data.data() + offset, sizeof(result));
    offset += sizeof(result);
    return result;
}

static size_t checked_size(
        uint64_t left, uint64_t right,
        const char * label) {
    if (left == 0 || right == 0 ||
            left > std::numeric_limits<size_t>::max() / right) {
        throw std::runtime_error(
            std::string("invalid MXFP8 ") + label);
    }
    return static_cast<size_t>(left * right);
}

} // namespace

namespace mxfp8 {

Data unpack(
        const std::vector<uint8_t> & blob) {
    constexpr size_t kHeaderBytes = 56;
    if (blob.size() < kHeaderBytes ||
            std::memcmp(blob.data(), "MXT1", 4) != 0 ||
            blob[4] != 1 || blob[5] != 8 ||
            blob[6] != 0 || blob[7] != 0) {
        throw std::runtime_error("invalid MXFP8 payload header");
    }
    size_t offset = 8;
    const uint64_t rows = u64(blob, offset);
    const uint64_t columns = u64(blob, offset);
    const uint64_t storage_rows = u64(blob, offset);
    const uint64_t storage_columns = u64(blob, offset);
    const uint64_t scale_rows = u64(blob, offset);
    const uint64_t scale_columns = u64(blob, offset);
    const bool block128=columns%128==0 && scale_rows==(rows+127)/128 && scale_columns==columns/128;
    const bool block32=columns%32==0 && scale_rows==(rows+31)/32 && scale_columns==columns/32;
    const bool row32=columns%32==0 && scale_rows==rows && scale_columns==columns/32;
    if (!rows || !columns || rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            columns > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            storage_rows != rows || storage_columns != columns || !(block128 || block32 || row32)) {
        throw std::runtime_error("invalid MXFP8 payload geometry");
    }
    const size_t value_bytes = checked_size(
        storage_rows, storage_columns, "value size");
    const size_t scale_bytes = checked_size(
        scale_rows, scale_columns, "scale size");
    if (value_bytes > blob.size() - offset ||
            scale_bytes != blob.size() - offset - value_bytes) {
        throw std::runtime_error("invalid MXFP8 payload length");
    }
    Data result;
    result.block_columns=block128?128:32;
    result.block_rows=block128?128:row32?1:32;
    result.out = static_cast<int64_t>(rows);
    result.neuron_len = static_cast<int64_t>(columns);
    result.values.assign(
        blob.begin() + static_cast<ptrdiff_t>(offset),
        blob.begin() + static_cast<ptrdiff_t>(offset + value_bytes));
    offset += value_bytes;
    result.scales.assign(
        blob.begin() + static_cast<ptrdiff_t>(offset),
        blob.end());
    if (std::any_of(
            result.values.begin(), result.values.end(),
            [](uint8_t value) { return (value & 0x7fu) == 0x7fu; })) {
        throw std::runtime_error("MXFP8 payload contains an E4M3 NaN code");
    }
    if (std::find(result.scales.begin(), result.scales.end(), 255u) !=
            result.scales.end()) {
        throw std::runtime_error("MXFP8 payload contains an E8M0 NaN scale");
    }
    return result;
}

} // namespace mxfp8

namespace mxfp4 {

Data unpack(
        const std::vector<uint8_t> & blob) {
    constexpr size_t kHeaderBytes = 56;
    if (blob.size() < kHeaderBytes ||
            std::memcmp(blob.data(), "MXT1", 4) != 0 ||
            blob[4] != 1 || blob[5] != 4 ||
            blob[6] != 0 || blob[7] != 0) {
        throw std::runtime_error("invalid MXFP4 payload header");
    }
    size_t offset = 8;
    const uint64_t rows = u64(blob, offset);
    const uint64_t columns = u64(blob, offset);
    const uint64_t storage_rows = u64(blob, offset);
    const uint64_t storage_columns = u64(blob, offset);
    const uint64_t scale_rows = u64(blob, offset);
    const uint64_t scale_columns = u64(blob, offset);
    if (rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            columns > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            columns % 32 != 0 || storage_rows != rows ||
            storage_columns != columns / 2 || scale_rows != rows ||
            scale_columns != columns / 32) {
        throw std::runtime_error("invalid MXFP4 payload geometry");
    }
    const size_t value_bytes = checked_size(
        storage_rows, storage_columns, "MXFP4 value size");
    const size_t scale_bytes = checked_size(
        scale_rows, scale_columns, "MXFP4 scale size");
    if (value_bytes > blob.size() - offset ||
            scale_bytes != blob.size() - offset - value_bytes) {
        throw std::runtime_error("invalid MXFP4 payload length");
    }
    Data result;
    result.out = static_cast<int64_t>(rows);
    result.neuron_len = static_cast<int64_t>(columns);
    result.values.assign(
        blob.begin() + static_cast<ptrdiff_t>(offset),
        blob.begin() + static_cast<ptrdiff_t>(offset + value_bytes));
    offset += value_bytes;
    result.scales.assign(
        blob.begin() + static_cast<ptrdiff_t>(offset), blob.end());
    if (std::find(result.scales.begin(), result.scales.end(), 255u) !=
            result.scales.end()) {
        throw std::runtime_error("MXFP4 payload contains an E8M0 NaN scale");
    }
    return result;
}

} // namespace mxfp4
} // namespace mfq
