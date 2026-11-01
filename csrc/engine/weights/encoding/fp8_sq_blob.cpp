#include "fp8_sq_blob.h"

#include "bytes.h"

namespace mfq::fp8sq {

std::vector<std::byte> block128_prefix(
    const std::vector<std::int64_t>& shape,
    std::string_view scale_dtype,
    const std::vector<std::int64_t>& scale_shape,
    const std::string& name) {
    if (shape.size() != 2 || scale_shape.size() != 2 ||
        shape[0] <= 0 || shape[1] <= 0 ||
        scale_shape != std::vector<std::int64_t>{
            (shape[0] + 127) / 128, (shape[1] + 127) / 128}) {
        throw std::runtime_error(
            "invalid native FP8-128 scale geometry: " + name);
    }
    ScaleKind scale_kind;
    if (scale_dtype == "BF16") {
        scale_kind = ScaleKind::Bf16;
    } else if (scale_dtype == "F16") {
        scale_kind = ScaleKind::F16;
    } else if (scale_dtype == "F32") {
        scale_kind = ScaleKind::F32;
    } else {
        throw std::runtime_error(
            "invalid native FP8-128 scale dtype: " + name);
    }

    std::vector<std::byte> result;
    const auto selector_bytes =
        ((static_cast<std::size_t>(shape[0]) * 3 + 7) / 8 + 3) &
        ~std::size_t{3};
    result.reserve(
        kHeaderBytes + selector_bytes +
        kPaletteBytes);
    for (const char value : std::string_view("F8SQ")) {
        result.push_back(static_cast<std::byte>(value));
    }
    result.push_back(std::byte{1});
    result.push_back(static_cast<std::byte>(scale_kind));
    encoding::append_little<std::uint16_t>(result, 0);
    encoding::append_little<std::uint16_t>(result, 128);
    encoding::append_little<std::uint16_t>(result, 128);
    encoding::append_little<std::uint64_t>(
        result, static_cast<std::uint64_t>(shape[0]));
    encoding::append_little<std::uint64_t>(
        result, static_cast<std::uint64_t>(shape[1]));
    encoding::append_little<std::uint64_t>(
        result, static_cast<std::uint64_t>(scale_shape[0]));
    encoding::append_little<std::uint64_t>(
        result, static_cast<std::uint64_t>(scale_shape[1]));

    const auto selector_offset = result.size();
    result.resize(selector_offset + selector_bytes, std::byte{0});
    for (std::size_t row = 0; row < static_cast<std::size_t>(shape[0]); ++row) {
        const auto bit = row * 3;
        const auto byte = selector_offset + bit / 8;
        const auto shift = static_cast<unsigned>(bit & 7);
        result[byte] |= static_cast<std::byte>(7u << shift);
        if (shift + 3 > 8) {
            result[byte + 1] |= static_cast<std::byte>(7u >> (8 - shift));
        }
    }

    const auto palette_offset = result.size();
    result.resize(palette_offset + kPaletteBytes, std::byte{0});
    std::size_t destination = palette_offset;
    for (unsigned bits = 1; bits <= 7; ++bits) {
        const auto count = std::size_t{1} << bits;
        for (unsigned code = 0, written = 0; written < count; ++code) {
            if ((code & 0x7Fu) == 0x7Fu) continue;
            result[destination++] = static_cast<std::byte>(code);
            ++written;
        }
    }
    return result;
}

} // namespace mfq::fp8sq
