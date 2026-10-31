#pragma once

#include "ggml.h"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

// Known GGUF row encodings. Backend capabilities are checked independently.
// Keep format dispatch independent of device names and model dimensions.
#define TYLOINFER_GGUF_LEGACY_FORMATS(X) \
    X(Q4_0) X(Q4_1) X(Q5_0) X(Q5_1) X(Q8_0) \
    X(Q2_K) X(Q3_K) X(Q4_K) X(Q5_K) X(Q6_K) X(Q8_K) \
    X(IQ2_XXS) X(IQ2_XS) X(IQ3_XXS) X(IQ4_NL) X(IQ3_S) \
    X(IQ2_S) X(IQ4_XS) X(Q2_0)

#define TYLOINFER_GGUF_COMPACT_FORMATS(X) \
    X(PTQ1_0) X(TQ2_T) X(TQK6) X(TQK7) X(TQP5) X(TQP6) X(TQP7) X(TQK5) X(TQK4)
#define TYLOINFER_GGUF_FORMATS(X) \
    TYLOINFER_GGUF_LEGACY_FORMATS(X) TYLOINFER_GGUF_COMPACT_FORMATS(X)

namespace mfq::gguf {
inline bool is_quantized(std::string_view dtype) noexcept {
    return dtype.substr(0, 5) == "GGUF_";
}

inline ggml_type quant_type(std::string_view dtype) {
#define TYLOINFER_GGUF_NAME(name) if (dtype == "GGUF_" #name) return GGML_TYPE_##name;
    TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_NAME)
#undef TYLOINFER_GGUF_NAME
    throw std::invalid_argument("unsupported GGUF quantization: " + std::string(dtype));
}

inline std::uint64_t row_bytes(ggml_type type, std::int64_t columns) {
    const auto block = ggml_blck_size(type);
    const auto bytes = ggml_type_size(type);
    if (columns <= 0 || block <= 0 || columns % block != 0 ||
            std::uint64_t(columns / block) > std::numeric_limits<std::uint64_t>::max() / bytes)
        throw std::invalid_argument("GGUF row width must contain complete quantization blocks");
    return std::uint64_t(columns / block) * bytes;
}
}
