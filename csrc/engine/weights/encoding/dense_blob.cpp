#include "dense_blob.h"

#include "bytes.h"

namespace tyloi::dense {

std::vector<std::byte> prefix(
    const std::vector<std::int64_t>& shape) {
    std::vector<std::byte> result;
    result.reserve(4 + shape.size() * 8);
    encoding::append_little<std::uint32_t>(
        result, static_cast<std::uint32_t>(shape.size()));
    for (const auto dimension : shape) {
        encoding::append_little<std::int64_t>(result, dimension);
    }
    return result;
}

} // namespace tyloi::dense
