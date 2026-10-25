#include "mfe_index.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mfq::mfe {
namespace {

template<class T>
T value(const std::vector<std::uint8_t>& data, std::size_t offset) {
    if (offset > data.size() || sizeof(T) > data.size() - offset)
        throw std::runtime_error("truncated MFE metadata");
    T result;
    std::memcpy(&result, data.data() + offset, sizeof(result));
    return result;
}

} // namespace

Index parse(std::size_t bytes,
            const std::function<void(std::size_t, std::uint8_t*, std::size_t)>& read) {
    if (!read || bytes < 20) throw std::runtime_error("invalid MFE header");
    std::size_t cursor = 0;
    const auto skip = [&](std::uint64_t size) {
        if (size > bytes - cursor) throw std::runtime_error("truncated MFE pool payload");
        const Part part{cursor, static_cast<std::size_t>(size)};
        cursor += part.bytes;
        return part;
    };
    const auto take = [&](std::size_t size) {
        const auto part = skip(size);
        std::vector<std::uint8_t> data(size);
        if (size) read(part.offset, data.data(), size);
        return data;
    };
    const auto header = take(20);
    Index result;
    if (!std::memcmp(header.data(), "NIM1", 4)) result.version = 1;
    else if (!std::memcmp(header.data(), "MFE1", 4) || !std::memcmp(header.data(), "NIM2", 4))
        result.version = 2;
    else if (!std::memcmp(header.data(), "MFD1", 4) || !std::memcmp(header.data(), "NID2", 4))
        throw std::runtime_error("MFE delta cannot be loaded as a full tensor");
    else throw std::runtime_error("invalid MFE header");

    const auto experts = value<std::uint32_t>(header, 4);
    const auto output = value<std::uint32_t>(header, 8);
    const auto width = value<std::uint32_t>(header, 12);
    const auto pools = value<std::uint32_t>(header, 16);
    if (!experts || !output || !width || !pools || pools > experts ||
        experts > std::numeric_limits<int>::max() || output > std::numeric_limits<int>::max() ||
        width > std::numeric_limits<int>::max() || experts > (bytes - 20) / sizeof(std::int32_t))
        throw std::runtime_error("invalid MFE dimensions");
    result.experts = static_cast<int>(experts);
    result.output = static_cast<int>(output);
    result.width = static_cast<int>(width);
    result.pools.reserve(pools);
    std::vector<bool> owned(experts, false);
    std::size_t seen = 0;
    for (std::uint32_t i = 0; i < pools; ++i) {
        const auto header = take(result.version == 1 ? 12 : 24);
        const auto count = value<std::uint32_t>(header, 0);
        const auto dtype_bytes = result.version == 1 ? 0 : value<std::uint32_t>(header, 4);
        const auto payload_bytes = value<std::uint64_t>(header, result.version == 1 ? 4 : 8);
        const auto runtime_bytes = result.version == 1 ? 0 : value<std::uint64_t>(header, 16);
        if (!count || count > experts || count > std::numeric_limits<int>::max() / output ||
            dtype_bytes > 32 || (result.version == 2 && !dtype_bytes) ||
            count > (std::numeric_limits<std::size_t>::max() - dtype_bytes) / sizeof(std::int32_t))
            throw std::runtime_error("invalid MFE pool dimensions");
        const auto metadata = take(std::size_t(count) * sizeof(std::int32_t) + dtype_bytes);
        Pool pool;
        pool.ids.resize(count);
        for (std::size_t j = 0; j < count; ++j) {
            const auto expert = value<std::int32_t>(metadata, j * sizeof(std::int32_t));
            if (expert < 0 || expert >= result.experts || owned[expert])
                throw std::runtime_error("invalid or duplicate MFE expert id");
            pool.ids[j] = expert;
            owned[expert] = true;
            ++seen;
        }
        pool.dtype = result.version == 1 ? "NINT" : std::string(
            reinterpret_cast<const char*>(metadata.data() + std::size_t(count) * sizeof(std::int32_t)),
            dtype_bytes);
        pool.runtime = skip(runtime_bytes);
        pool.payload = skip(payload_bytes);
        result.pools.push_back(std::move(pool));
    }
    if (cursor != bytes || seen != experts)
        throw std::runtime_error("MFE expert coverage or tail mismatch");
    return result;
}

} // namespace mfq::mfe
