#pragma once

#include "backend.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace tq_detail {

inline int dtype(mfq_tensor_backend::ScalarType type) {
    if (type == mfq_tensor_backend::kFloat16) return 0;
    if (type == mfq_tensor_backend::kBFloat16) return 1;
    if (type == mfq_tensor_backend::kFloat32) return 2;
    throw std::invalid_argument("TQ activation dtype must be FP16, BF16 or FP32");
}

inline void status(int code) {
    if (code < 0) throw std::invalid_argument("invalid TQ CUDA projection arguments");
    if (code >= 1000)
        throw std::runtime_error("TQ cuBLAS status " + std::to_string(code - 1000));
    TYLOINFER_CUDA_CHECK(static_cast<cudaError_t>(code));
}

inline int gemm_min_tokens(mfq_tensor_backend::ScalarType type) {
    return type == mfq_tensor_backend::kFloat16 ? 32 : 128;
}

} // namespace tq_detail
