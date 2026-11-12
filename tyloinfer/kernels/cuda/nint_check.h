#pragma once

#include <torch/extension.h>

namespace mfq_jit_nint {

/** Validate the packed NINT8-zero weight and scale tensors. */
inline void validate_nint8_zero(
        const torch::Tensor & quantized, const torch::Tensor & scale) {
    TORCH_CHECK(
        quantized.is_cuda() && quantized.is_contiguous() &&
        quantized.scalar_type() == at::kByte &&
        quantized.dim() == 3 && quantized.size(2) == 32,
        "NINT8-0 q must be contiguous CUDA uint8 [N,G,32]");
    TORCH_CHECK(
        scale.is_cuda() && scale.is_contiguous() &&
        scale.scalar_type() == at::kHalf &&
        scale.dim() == 2 && scale.size(0) == quantized.size(0) &&
        scale.size(1) == quantized.size(1),
        "NINT8-0 scale must be contiguous CUDA fp16 [N,G]");
    TORCH_CHECK(
        quantized.device() == scale.device(),
        "NINT8-0 tensors must share one CUDA device");
}

} // namespace mfq_jit_nint
