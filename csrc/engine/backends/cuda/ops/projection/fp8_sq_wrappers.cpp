#include "backend.h"
#include "fp8_sq_raw.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
namespace raw = ::tyloi::cuda::fp8_sq_raw;
namespace {
using Layout = raw::Layout;
constexpr int kDirectPackedMaxRows=48;
void validate_metadata(
        const ::tyloi::DenseTensor& blob,
        const ::tyloi::DenseTensor& row_q,
        const ::tyloi::DenseTensor& row_symbol_byte_offsets,
        const Layout& layout,
        bool mxfp8) {
    TYLOINFER_RUNTIME_CHECK(
        blob.is_cuda() && blob.scalar_type() == ::tyloi::kUInt8 &&
        blob.dim() == 1 && blob.is_contiguous(),
        "FP8-SQ blob must be contiguous rank-1 CUDA uint8");
    TYLOINFER_RUNTIME_CHECK(
        row_q.is_cuda() && row_q.get_device() == blob.get_device() &&
        row_q.scalar_type() == ::tyloi::kUInt8 &&
        row_q.dim() == 1 && row_q.is_contiguous() &&
        row_q.numel() == layout.outputs &&
        row_symbol_byte_offsets.is_cuda() &&
        row_symbol_byte_offsets.get_device() == blob.get_device() &&
        row_symbol_byte_offsets.scalar_type() == ::tyloi::kInt32 &&
        row_symbol_byte_offsets.dim() == 1 &&
        row_symbol_byte_offsets.is_contiguous() &&
        row_symbol_byte_offsets.numel() == layout.outputs,
        "FP8-SQ row metadata must be matching CUDA arrays");
    TYLOINFER_RUNTIME_CHECK(
        layout.outputs > 0 && layout.width > 0 &&
        layout.block_rows > 0 && layout.block_columns > 0 &&
        layout.scale_rows == (layout.outputs + layout.block_rows - 1) /
            layout.block_rows &&
        layout.scale_columns == (layout.width + layout.block_columns - 1) /
            layout.block_columns &&
        layout.palettes < layout.symbols && layout.symbols <= layout.scales &&
        layout.scales < static_cast<std::size_t>(blob.numel()),
        "FP8-SQ layout metadata is inconsistent");
    const std::size_t scale_count =
        static_cast<std::size_t>(layout.scale_rows) * layout.scale_columns;
    const std::size_t scale_itemsize = mxfp8 || layout.scale_kind == 1
        ? 1
        : layout.scale_kind == 4 ? 4 : 2;
    TYLOINFER_RUNTIME_CHECK(
        scale_count <=
            (static_cast<std::size_t>(blob.numel()) - layout.scales) /
                scale_itemsize,
        "FP8-SQ scale payload exceeds its blob");
    if (mxfp8) {
        TYLOINFER_RUNTIME_CHECK(
            layout.scale_kind == 1 &&
            ((layout.block_rows == 1 && layout.block_columns == 32) ||
             (layout.block_rows == 32 && layout.block_columns == 32) ||
             (layout.block_rows == 128 && layout.block_columns == 128)),
            "MXFP8-SQ scale contract is invalid");
    } else {
        TYLOINFER_RUNTIME_CHECK(
            layout.block_rows == 128 && layout.block_columns == 128 &&
            (layout.scale_kind == 2 || layout.scale_kind == 3 ||
             layout.scale_kind == 4),
            "FP8-128SQ scale contract is invalid");
    }
}

Layout make_layout(
        std::int64_t outputs,
        std::int64_t width,
        std::int64_t block_rows,
        std::int64_t block_columns,
        std::int64_t scale_rows,
        std::int64_t scale_columns,
        std::int64_t scale_kind,
        std::int64_t palettes,
        std::int64_t symbols,
        std::int64_t scales) {
    TYLOINFER_RUNTIME_CHECK(
        outputs > 0 && outputs <= std::numeric_limits<int>::max() &&
        width > 0 && width <= std::numeric_limits<int>::max() &&
        block_rows > 0 && block_rows <= std::numeric_limits<int>::max() &&
        block_columns > 0 && block_columns <= std::numeric_limits<int>::max() &&
        scale_rows > 0 && scale_rows <= std::numeric_limits<int>::max() &&
        scale_columns > 0 && scale_columns <= std::numeric_limits<int>::max() &&
        palettes >= 0 && symbols >= 0 && scales >= 0,
        "FP8-SQ host layout exceeds CUDA limits");
    return {
        static_cast<int>(outputs),
        static_cast<int>(width),
        static_cast<int>(block_rows),
        static_cast<int>(block_columns),
        static_cast<int>(scale_rows),
        static_cast<int>(scale_columns),
        static_cast<int>(scale_kind),
        static_cast<std::size_t>(palettes),
        static_cast<std::size_t>(symbols),
        static_cast<std::size_t>(scales),
    };
}

template <bool MXFP8>
::tyloi::DenseTensor dequant(
        ::tyloi::DenseTensor blob,
        ::tyloi::DenseTensor row_q,
        ::tyloi::DenseTensor row_symbol_byte_offsets,
        Layout layout,
        bool fp32) {
    validate_metadata(blob, row_q, row_symbol_byte_offsets, layout, MXFP8);
    const MfqCudaGuard guard(blob.device());
    auto output = ::tyloi::cuda::empty(
        {layout.outputs, layout.width},
        blob.options().dtype(
            fp32 ? ::tyloi::kFloat32 : ::tyloi::kFloat16));
    const auto stream = mfq_current_cuda_stream();
    raw::dequant(blob.data_ptr<std::uint8_t>(),row_q.data_ptr<std::uint8_t>(),row_symbol_byte_offsets.data_ptr<std::int32_t>(),output.data_ptr(),layout,MXFP8,fp32,stream);
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

void launch_dense_gemm_nt(const ::tyloi::DenseTensor& input,const ::tyloi::DenseTensor& weight,
        ::tyloi::DenseTensor& output,cudaStream_t stream) {
    auto handle=mfq_current_cublas_handle();
    TYLOINFER_RUNTIME_CHECK(cublasSetStream(handle,stream)==CUBLAS_STATUS_SUCCESS,"FP8-SQ cublasSetStream failed");
    const bool half=input.scalar_type()==::tyloi::kFloat16;
    TYLOINFER_RUNTIME_CHECK(raw::gemm(handle,input.data_ptr(),weight.data_ptr(),output.data_ptr(),int(input.size(0)),int(input.size(1)),int(weight.size(0)),half?::tyloi::cuda::NumericType::f16 : ::tyloi::cuda::NumericType::f32,stream)==CUBLAS_STATUS_SUCCESS,
        half?"FP8-SQ FP16 GEMM failed":"FP8-SQ FP32 GEMM failed");
}
template <bool MXFP8>
::tyloi::DenseTensor matmul(
        ::tyloi::DenseTensor blob,
        ::tyloi::DenseTensor row_q,
        ::tyloi::DenseTensor row_symbol_byte_offsets,
        ::tyloi::DenseTensor input,
        Layout layout) {
    validate_metadata(blob, row_q, row_symbol_byte_offsets, layout, MXFP8);
    TYLOINFER_RUNTIME_CHECK(
        input.is_cuda() && input.get_device() == blob.get_device() &&
        input.dim() == 2 && input.is_contiguous() &&
        input.size(1) == layout.width &&
        input.size(0) <= std::numeric_limits<int>::max() &&
        (input.scalar_type() == ::tyloi::kFloat16 ||
         input.scalar_type() == ::tyloi::kFloat32),
        "FP8-SQ activation must be contiguous rank-2 CUDA FP16/FP32");
    const MfqCudaGuard guard(blob.device());
    ::tyloi::DenseTensor output = ::tyloi::cuda::empty(
        {input.size(0), layout.outputs}, input.options());
    const int rows = static_cast<int>(input.size(0));
    if (rows == 0) return output;
    const auto stream = mfq_current_cuda_stream();
    if (rows > kDirectPackedMaxRows) {
        // The decoded matrix is a transient workspace owned by this call.  It
        // is never cached or attached to the packed weight.
        auto weight = dequant<MXFP8>(
            blob, row_q, row_symbol_byte_offsets, layout,
            input.scalar_type() == ::tyloi::kFloat32);
        launch_dense_gemm_nt(input, weight, output, stream);
        TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
    }
    raw::matmul(blob.data_ptr<std::uint8_t>(),row_q.data_ptr<std::uint8_t>(),row_symbol_byte_offsets.data_ptr<std::int32_t>(),input.data_ptr(),output.data_ptr(),layout,rows,MXFP8,input.scalar_type()==::tyloi::kFloat32 ? ::tyloi::cuda::NumericType::f32 : ::tyloi::cuda::NumericType::f16,stream);
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template <bool MXFP8>
void routed_matmul(
        ::tyloi::DenseTensor blob,
        ::tyloi::DenseTensor row_q,
        ::tyloi::DenseTensor row_symbol_byte_offsets,
        ::tyloi::DenseTensor input,
        ::tyloi::DenseTensor expert_ids,
        ::tyloi::DenseTensor expert_local,
        std::int64_t n_experts,
        std::int64_t local_experts,
        std::int64_t out_per_expert,
        Layout layout,
        ::tyloi::DenseTensor output) {
    validate_metadata(blob, row_q, row_symbol_byte_offsets, layout, MXFP8);
    TYLOINFER_RUNTIME_CHECK(
        layout.outputs == local_experts * out_per_expert &&
        input.is_cuda() && input.get_device() == blob.get_device() &&
        input.is_contiguous() && input.scalar_type() == ::tyloi::kFloat16 &&
        (input.dim() == 2 || input.dim() == 3) && input.size(-1) == layout.width &&
        expert_ids.is_cuda() && expert_ids.get_device() == blob.get_device() &&
        expert_ids.is_contiguous() &&
        expert_ids.scalar_type() == ::tyloi::kInt32 &&
        expert_ids.dim() == 2 && expert_ids.size(0) == input.size(0) &&
        expert_local.is_cuda() && expert_local.get_device() == blob.get_device() &&
        expert_local.is_contiguous() &&
        expert_local.scalar_type() == ::tyloi::kInt32 &&
        expert_local.dim() == 1 && expert_local.numel() == n_experts &&
        output.is_cuda() && output.get_device() == blob.get_device() &&
        output.is_contiguous() && output.scalar_type() == ::tyloi::kFloat16 &&
        output.sizes() == ::tyloi::IntArrayRef(
            {expert_ids.size(0), expert_ids.size(1), out_per_expert}) &&
        (input.dim() == 2 || input.size(1) == expert_ids.size(1)) &&
        n_experts > 0 && n_experts <= std::numeric_limits<int>::max() &&
        local_experts > 0 && local_experts <= std::numeric_limits<int>::max() &&
        out_per_expert > 0 && out_per_expert <= std::numeric_limits<int>::max(),
        "FP8-SQ routed matmul requires matching contiguous CUDA FP16 tensors");
    TYLOINFER_RUNTIME_CHECK(
        expert_ids.numel() <= std::numeric_limits<int>::max(),
        "FP8-SQ route count exceeds CUDA limits");
    if (expert_ids.numel() == 0) return;
    const MfqCudaGuard guard(blob.device());
    raw::routed(blob.data_ptr<std::uint8_t>(),row_q.data_ptr<std::uint8_t>(),row_symbol_byte_offsets.data_ptr<std::int32_t>(),input.data_ptr(),output.data_ptr(),expert_ids.data_ptr<std::int32_t>(),expert_local.data_ptr<std::int32_t>(),layout,int(expert_ids.numel()),int(expert_ids.size(1)),int(n_experts),int(local_experts),int(out_per_expert),input.dim()==2,MXFP8,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
}

}
::tyloi::DenseTensor mxfp8_sq_dequant_cuda(
        ::tyloi::DenseTensor blob,
        ::tyloi::DenseTensor row_q,
        ::tyloi::DenseTensor row_symbol_byte_offsets,
        std::int64_t outputs, std::int64_t width,
        std::int64_t block_rows, std::int64_t block_columns,
        std::int64_t scale_rows, std::int64_t scale_columns,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset, bool fp32) {
    return dequant<true>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets),
        make_layout(outputs, width, block_rows, block_columns,
                    scale_rows, scale_columns, 1,
                    palettes_offset, symbols_offset, scales_offset),
        fp32);
}

::tyloi::DenseTensor fp8_128_sq_dequant_cuda(
        ::tyloi::DenseTensor blob,
        ::tyloi::DenseTensor row_q,
        ::tyloi::DenseTensor row_symbol_byte_offsets,
        std::int64_t outputs, std::int64_t width,
        std::int64_t scale_kind,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset, bool fp32) {
    return dequant<false>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets),
        make_layout(outputs, width, 128, 128,
                    (outputs + 127) / 128, (width + 127) / 128,
                    scale_kind, palettes_offset, symbols_offset, scales_offset),
        fp32);
}

::tyloi::DenseTensor mxfp8_sq_matmul_cuda(
        ::tyloi::DenseTensor blob,
        ::tyloi::DenseTensor row_q,
        ::tyloi::DenseTensor row_symbol_byte_offsets,
        ::tyloi::DenseTensor input,
        std::int64_t outputs, std::int64_t width,
        std::int64_t block_rows, std::int64_t block_columns,
        std::int64_t scale_rows, std::int64_t scale_columns,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset) {
    return matmul<true>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets), std::move(input),
        make_layout(outputs, width, block_rows, block_columns,
                    scale_rows, scale_columns, 1,
                    palettes_offset, symbols_offset, scales_offset));
}

::tyloi::DenseTensor fp8_128_sq_matmul_cuda(
        ::tyloi::DenseTensor blob,
        ::tyloi::DenseTensor row_q,
        ::tyloi::DenseTensor row_symbol_byte_offsets,
        ::tyloi::DenseTensor input,
        std::int64_t outputs, std::int64_t width,
        std::int64_t scale_kind,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset) {
    return matmul<false>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets), std::move(input),
        make_layout(outputs, width, 128, 128,
                    (outputs + 127) / 128, (width + 127) / 128,
                    scale_kind, palettes_offset, symbols_offset, scales_offset));
}

void mxfp8_sq_moe_matmul_cuda(
        ::tyloi::DenseTensor blob,
        ::tyloi::DenseTensor row_q,
        ::tyloi::DenseTensor row_symbol_byte_offsets,
        ::tyloi::DenseTensor input,
        ::tyloi::DenseTensor expert_ids,
        ::tyloi::DenseTensor expert_local,
        std::int64_t n_experts, std::int64_t local_experts,
        std::int64_t out_per_expert, std::int64_t width,
        std::int64_t block_rows, std::int64_t block_columns,
        std::int64_t scale_rows, std::int64_t scale_columns,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset,
        ::tyloi::DenseTensor output) {
    routed_matmul<true>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets), std::move(input),
        std::move(expert_ids), std::move(expert_local),
        n_experts, local_experts, out_per_expert,
        make_layout(local_experts * out_per_expert, width,
                    block_rows, block_columns, scale_rows, scale_columns, 1,
                    palettes_offset, symbols_offset, scales_offset),
        std::move(output));
}

void fp8_128_sq_moe_matmul_cuda(
        ::tyloi::DenseTensor blob,
        ::tyloi::DenseTensor row_q,
        ::tyloi::DenseTensor row_symbol_byte_offsets,
        ::tyloi::DenseTensor input,
        ::tyloi::DenseTensor expert_ids,
        ::tyloi::DenseTensor expert_local,
        std::int64_t n_experts, std::int64_t local_experts,
        std::int64_t out_per_expert, std::int64_t width,
        std::int64_t scale_kind,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset,
        ::tyloi::DenseTensor output) {
    routed_matmul<false>(
        std::move(blob), std::move(row_q),
        std::move(row_symbol_byte_offsets), std::move(input),
        std::move(expert_ids), std::move(expert_local),
        n_experts, local_experts, out_per_expert,
        make_layout(local_experts * out_per_expert, width, 128, 128,
                    (local_experts * out_per_expert + 127) / 128,
                    (width + 127) / 128, scale_kind,
                    palettes_offset, symbols_offset, scales_offset),
        std::move(output));
}
