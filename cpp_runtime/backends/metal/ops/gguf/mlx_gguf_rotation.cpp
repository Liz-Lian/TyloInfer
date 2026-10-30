#include "mlx_gguf.h"
#include "mlx_gguf_rotation.h"
#include "mfq_container.h"
#include "mlx_weight_residency.h"
#include "nlohmann/json.hpp"

#include <cmath>
#include <climits>
#include <stdexcept>

namespace mfq::metal {
using namespace mlx::core;
namespace {
array rotate128(const array& input, const MlxGgufRotation& rotation, int width, bool inverse, Dtype dtype) {
    static const auto kernel = fast::metal_kernel("tinfer_gguf_hadamard128", {"x", "signs", "params"}, {"y"}, R"METAL(
uint block = thread_position_in_grid.x / 32u, lane = thread_index_in_simdgroup;
if (block >= params[0]) return;
uint at = block * 128u + lane * 4u, col = at % params[1];
float4 value(x[at], x[at + 1u], x[at + 2u], x[at + 3u]);
float4 sign = gguf_rotation_signs<SIGNS>(signs, col);
if constexpr (!INVERSE) value *= sign;
value = float4(value.x + value.y, value.x - value.y, value.z + value.w, value.z - value.w);
value = float4(value.x + value.z, value.y + value.w, value.x - value.z, value.y - value.w);
for (ushort mask = 1u; mask < 32u; mask *= 2u) {
    float4 other(simd_shuffle_xor(value.x, mask), simd_shuffle_xor(value.y, mask),
        simd_shuffle_xor(value.z, mask), simd_shuffle_xor(value.w, mask));
    value = lane & mask ? other - value : value + other;
}
value *= 0.08838834764831845f;
if constexpr (INVERSE) value *= sign;
y[at] = T(value.x); y[at + 1u] = T(value.y); y[at + 2u] = T(value.z); y[at + 3u] = T(value.w);
)METAL", R"METAL(
template<bool ENABLED, typename S>
float4 gguf_rotation_signs(S signs, uint col) {
    if constexpr (ENABLED) return float4(signs[col], signs[col + 1u], signs[col + 2u], signs[col + 3u]);
    else return float4(1.f);
}
)METAL");
    if (input.size() > INT_MAX) throw std::overflow_error("GGUF Hadamard activation extent is too large");
    const auto blocks = int(input.size() / 128);
    if (!blocks) return astype(input, dtype);
    return kernel({contiguous(input), rotation.signs, array({uint32_t(blocks), uint32_t(width)})},
        {input.shape()}, {dtype}, {((blocks + 3) / 4) * 128, 1, 1}, {128, 1, 1},
        {{"T", dtype}, {"SIGNS", !rotation.pattern.empty()}, {"INVERSE", inverse}}, std::nullopt, false, {}).front();
}
}

void MlxGgufWeight::configure_hadamard(const MfqContainer& model, const std::string& name) {
    const auto found = model.header().extra_json.find("gguf_hadamard");
    if (found == model.header().extra_json.end()) return;
    const auto rotations = nlohmann::json::parse(found->second);
    if (!rotations.contains(name)) return;
    const auto& entry = rotations.at(name);
    if (planes_ || native_) throw std::invalid_argument("GGUF Hadamard must be configured before layout preparation");
    const int block = entry.at("block_size").get<int>();
    const auto pattern = entry.at("signs").get<std::vector<float>>();
    if (block < 1 || (block & (block - 1)) || input_size() % block ||
        (!pattern.empty() && pattern.size() != std::size_t(input_size())))
        throw std::invalid_argument("invalid GGUF Hadamard geometry");
    for (const auto value : pattern) if (value != -1.f && value != 1.f)
        throw std::invalid_argument("invalid GGUF Hadamard sign");
    auto signs = pattern.empty() ? array(1.f) : array(pattern.begin(), Shape{input_size()}, float32);
    MlxWeightResidency::track(signs);
    rotation_ = std::make_shared<MlxGgufRotation>(MlxGgufRotation{block, pattern, std::move(signs)});
}

bool MlxGgufWeight::same_rotation(const MlxGgufWeight& other) const {
    return rotation_ == other.rotation_ || (rotation_ && other.rotation_ && rotation_->block == other.rotation_->block &&
        rotation_->pattern == other.rotation_->pattern);
}

array MlxGgufWeight::rotate_input(const array& input) const {
    if (rotation_->block == 128) return rotate128(input, *rotation_, input_size(), false, input.dtype());
    const auto rows = input.size() / rotation_->block;
    if (rows > INT_MAX) throw std::overflow_error("GGUF Hadamard activation extent is too large");
    auto signed_input = reshape(astype(input, float32), {-1, input_size()}) * rotation_->signs;
    auto result = hadamard_transform(reshape(signed_input, {int(rows), rotation_->block}),
        1.f / std::sqrt(float(rotation_->block)));
    return astype(reshape(result, input.shape()), input.dtype());
}

array MlxGgufWeight::unrotate_rows(const array& input, Dtype dtype) const {
    if (rotation_->block == 128) return rotate128(input, *rotation_, input_size(), true, dtype);
    const auto rows = input.size() / rotation_->block;
    if (rows > INT_MAX) throw std::overflow_error("GGUF Hadamard weight extent is too large");
    auto result = hadamard_transform(reshape(astype(input, float32), {int(rows), rotation_->block}),
        1.f / std::sqrt(float(rotation_->block)));
    result = reshape(result, {-1, input_size()}) * rotation_->signs;
    return astype(reshape(result, input.shape()), dtype);
}
}
