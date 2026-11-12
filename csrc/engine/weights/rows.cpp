#include "engine/weights/rows.h"
#include "engine/weights/source/source.h"
#include "engine/ops.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace tyloi::weights {

Tensor read_rows(const source::Source& source, std::string_view name,
        std::int64_t first, std::int64_t count) {
    const auto* record = source.find_tensor(name);
    if (!record || !record->shape || record->shape->size() != 2)
        throw std::runtime_error("reference requires a matrix: " + std::string(name));
    const auto& shape = *record->shape;
    if (first < 0 || count < 1 || shape[0] < count || first > shape[0] - count || shape[1] < 1)
        throw std::runtime_error("reference matrix row range is invalid");
    ScalarType dtype;
    if (record->dtype == "F16") dtype = kFloat16;
    else if (record->dtype == "BF16") dtype = kBFloat16;
    else if (record->dtype == "F32") dtype = kFloat32;
    else throw std::runtime_error("reference embedding/output matrix requires a floating source: " + std::string(name));
    constexpr auto header = sizeof(std::uint32_t) + 2 * sizeof(std::int64_t);
    const auto element = scalar_size(dtype);
    if (std::uint64_t(shape[1]) > (std::numeric_limits<std::uint64_t>::max() - header) / element)
        throw std::overflow_error("reference matrix row byte count overflows");
    const auto row_bytes = std::uint64_t(shape[1]) * element;
    if (std::uint64_t(first) > (std::numeric_limits<std::uint64_t>::max() - header) / row_bytes ||
        std::uint64_t(count) > std::numeric_limits<std::size_t>::max() / row_bytes)
        throw std::overflow_error("reference matrix range overflows");
    const auto offset = header + std::uint64_t(first) * row_bytes;
    const auto bytes = std::size_t(count) * row_bytes;
    if (offset > record->nbytes || bytes > record->nbytes - offset)
        throw std::runtime_error("reference matrix row range is truncated");
    auto result = ops::empty({count, shape[1]}, TensorOptions().device(kCPU).dtype(dtype));
    source.read_range_into(name, offset, reinterpret_cast<std::byte*>(result.dense().data_ptr()), bytes);
    return result;
}

} // namespace tyloi::weights
