#include "fp8_sq.h"
#include "fp8_sq_raw.h"
#include "fp8_sq_check.h"
#include "packed_backward.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <limits>
#include <utility>
namespace {
using Layout=tyloi::cuda::fp8_sq_raw::Layout;
using mfq_jit_fp8::validate_metadata;
using mfq_jit_fp8::make_layout;
template <bool MXFP8>
torch::Tensor backward_input(
        torch::Tensor blob,
        torch::Tensor row_q,
        torch::Tensor row_symbol_byte_offsets,
        torch::Tensor output_gradient,
        Layout layout) {
    validate_metadata(blob, row_q, row_symbol_byte_offsets, layout, MXFP8);
    TORCH_CHECK(
        output_gradient.is_cuda() &&
        output_gradient.get_device() == blob.get_device() &&
        output_gradient.dim() == 2 && output_gradient.is_contiguous() &&
        output_gradient.size(1) == layout.outputs &&
        output_gradient.size(0) <= std::numeric_limits<int>::max() &&
        (output_gradient.scalar_type() == at::kHalf ||
         output_gradient.scalar_type() == at::kBFloat16 ||
         output_gradient.scalar_type() == at::kFloat),
        "FP8-SQ backward requires contiguous CUDA FP16/BF16/FP32 [M,N]");
    const c10::cuda::CUDAGuard guard(blob.device());
    auto result = torch::empty(
        {output_gradient.size(0), layout.width}, output_gradient.options());
    if (output_gradient.size(0) == 0) return result;
    torch::Tensor weight;
    if constexpr(MXFP8) weight=mxfp8_sq_dequant_cuda(blob,row_q,row_symbol_byte_offsets,layout.outputs,layout.width,
        layout.block_rows,layout.block_columns,layout.scale_rows,layout.scale_columns,layout.palettes,layout.symbols,layout.scales,false);
    else weight=fp8_128_sq_dequant_cuda(blob,row_q,row_symbol_byte_offsets,layout.outputs,layout.width,layout.scale_kind,
        layout.palettes,layout.symbols,layout.scales,false);
    mfq_jit_backward::launch_dense_half_weight(
        output_gradient, weight, result,
        static_cast<int>(output_gradient.size(0)),
        layout.outputs, layout.width, at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

}
torch::Tensor mxfp8_sq_backward_input_cuda(
        torch::Tensor blob,
        torch::Tensor row_q,
        torch::Tensor row_symbol_byte_offsets,
        torch::Tensor output_gradient,
        std::int64_t outputs, std::int64_t width,
        std::int64_t block_rows, std::int64_t block_columns,
        std::int64_t scale_rows, std::int64_t scale_columns,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset) {
    return backward_input<true>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets), std::move(output_gradient),
        make_layout(outputs, width, block_rows, block_columns,
                    scale_rows, scale_columns, 1,
                    palettes_offset, symbols_offset, scales_offset));
}

torch::Tensor fp8_128_sq_backward_input_cuda(
        torch::Tensor blob,
        torch::Tensor row_q,
        torch::Tensor row_symbol_byte_offsets,
        torch::Tensor output_gradient,
        std::int64_t outputs, std::int64_t width,
        std::int64_t scale_kind,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset) {
    return backward_input<false>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets), std::move(output_gradient),
        make_layout(outputs, width, 128, 128,
                    (outputs + 127) / 128, (width + 127) / 128,
                    scale_kind, palettes_offset, symbols_offset, scales_offset));
}
