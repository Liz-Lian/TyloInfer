#include "tyloinfer/mlx_affine.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace mfq {
namespace {
constexpr char magic[] = "MLXA";
std::uint64_t product(std::uint64_t left, std::uint64_t right) {
    if (left && right > std::numeric_limits<std::uint64_t>::max() / left)
        throw std::overflow_error("MLX affine tensor extent overflow");
    return left * right;
}
}

void MlxAffineLayout::validate() {
    if ((shape.size() != 2 && shape.size() != 3) ||
        std::any_of(shape.begin(), shape.end(), [](auto d) { return d < 1; }) ||
        (bits != 2 && bits != 3 && bits != 4 && bits != 5 && bits != 6 && bits != 8) ||
        (group_size != 32 && group_size != 64 && group_size != 128) ||
        (scale_dtype != "BF16" && scale_dtype != "F16" && scale_dtype != "F32"))
        throw std::invalid_argument("unsupported MLX affine tensor layout");
    if (shape.back() % 32 || shape.back() % group_size)
        throw std::invalid_argument("MLX affine row is not group aligned");
    std::uint64_t rows = 1;
    for (std::size_t i = 0; i + 1 < shape.size(); ++i) rows = product(rows, shape[i]);
    values_bytes = product(product(rows, shape.back() / 32), bits * 4u);
    scales_bytes = product(product(rows, shape.back() / group_size), scale_dtype == "F32" ? 4u : 2u);
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    if (prefix_bytes > maximum - values_bytes || scales_bytes > (maximum - prefix_bytes - values_bytes) / 2)
        throw std::overflow_error("MLX affine tensor extent overflow");
}
std::vector<std::byte> MlxAffineLayout::prefix() const {
    auto checked = *this;
    checked.validate();
    const auto payload = nlohmann::json{{"shape", shape}, {"bits", bits}, {"group_size", group_size},
        {"scale_dtype", scale_dtype}}.dump();
    if (payload.size() > 4088) throw std::invalid_argument("MLX affine metadata is too large");
    std::vector<std::byte> bytes(8 + payload.size());
    std::memcpy(bytes.data(), magic, 4);
    for (unsigned i = 0; i < 4; ++i) bytes[4 + i] = std::byte(payload.size() >> (8 * i));
    std::memcpy(bytes.data() + 8, payload.data(), payload.size());
    return bytes;
}
std::size_t MlxAffineLayout::prefix_size(const std::byte* bytes, std::size_t size) {
    if (size < 8 || !bytes || std::memcmp(bytes, magic, 4))
        throw std::invalid_argument("invalid MLX affine metadata");
    std::uint32_t count = 0;
    for (unsigned i = 0; i < 4; ++i) count |= std::uint32_t(bytes[4 + i]) << (8 * i);
    if (!count || count > 4088) throw std::invalid_argument("invalid MLX affine metadata length");
    return 8u + count;
}
MlxAffineLayout MlxAffineLayout::parse(const std::byte* bytes, std::size_t size) {
    const auto count = prefix_size(bytes, size);
    if (size < count) throw std::invalid_argument("truncated MLX affine metadata");
    const auto config = nlohmann::json::parse(reinterpret_cast<const char*>(bytes + 8),
        reinterpret_cast<const char*>(bytes + count));
    MlxAffineLayout result;
    result.shape = config.at("shape").get<std::vector<std::int64_t>>();
    result.bits = config.at("bits").get<int>();
    result.group_size = config.at("group_size").get<int>();
    result.scale_dtype = config.at("scale_dtype").get<std::string>();
    result.prefix_bytes = count;
    result.validate();
    return result;
}

} // namespace mfq
