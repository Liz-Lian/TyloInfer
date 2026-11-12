#pragma once
#include <torch/extension.h>
#include "mxfp4_sq_raw.h"
#include <cstdint>
namespace mfq_jit_sq4 {
/** Validate the MXFP4-SQ payload and construct its packed layout. */
inline tyloi::sq::Layout validate(
        const torch::Tensor& blob,
        const torch::Tensor& row_q,
        const torch::Tensor& row_symbol_byte_offsets,
        const torch::Tensor& row_auxiliary,
        std::int64_t bits,
        std::int64_t outputs,
        std::int64_t width,
        std::int64_t base,
        std::int64_t q_sum,
        std::int64_t sq4_rows) {
    const auto q = bits == 0
        ? tyloi::sq::adaptive_layout(outputs, width, base, q_sum, sq4_rows)
        : tyloi::sq::layout(bits, outputs, width, base);
    TORCH_CHECK(blob.is_cuda() && blob.scalar_type() == at::kByte &&
        blob.dim() == 1 && blob.is_contiguous() && std::size_t(blob.numel()) == q.bytes,
        "MXFP4-SQ blob requires contiguous rank-1 CUDA uint8 and exact payload length");
    TORCH_CHECK(
        row_q.is_cuda() && row_q.get_device() == blob.get_device() &&
        row_q.scalar_type() == at::kByte &&
        row_q.dim() == 1 && row_q.is_contiguous() && row_q.numel() == outputs &&
        row_symbol_byte_offsets.is_cuda() &&
        row_symbol_byte_offsets.get_device() == blob.get_device() &&
        row_symbol_byte_offsets.scalar_type() == at::kInt &&
        row_symbol_byte_offsets.dim() == 1 &&
        row_symbol_byte_offsets.is_contiguous() &&
        row_symbol_byte_offsets.numel() == outputs &&
        row_auxiliary.is_cuda() && row_auxiliary.get_device() == blob.get_device() &&
        row_auxiliary.scalar_type() == at::kInt &&
        row_auxiliary.dim() == 1 && row_auxiliary.is_contiguous() &&
        row_auxiliary.numel() == outputs,
        "MXFP4-SQ row metadata requires matching contiguous CUDA arrays");
    TORCH_CHECK(reinterpret_cast<std::uintptr_t>(blob.data_ptr<std::uint8_t>()) % 4 == 0,
        "MXFP4-SQ blob must be 4-byte aligned");
    return q;
}
} // namespace mfq_jit_sq4
