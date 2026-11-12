#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "fp8_sq.h"
#include "fp8_sq_raw.h"
#include "fp8_sq_check.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
namespace raw = ::tyloi::cuda::fp8_sq_raw;
namespace {
using Layout = raw::Layout;
constexpr int kDirectPackedMaxRows=48;
using mfq_jit_fp8::validate_metadata;
using mfq_jit_fp8::make_layout;
template <bool MXFP8>
torch::Tensor dequant(
        torch::Tensor blob,
        torch::Tensor row_q,
        torch::Tensor row_symbol_byte_offsets,
        Layout layout,
        bool fp32) {
    validate_metadata(blob, row_q, row_symbol_byte_offsets, layout, MXFP8);
    const c10::cuda::CUDAGuard guard(blob.device());
    auto output = torch::empty(
        {layout.outputs, layout.width},
        blob.options().dtype(
            fp32 ? at::kFloat : at::kHalf));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    raw::dequant(blob.data_ptr<std::uint8_t>(),row_q.data_ptr<std::uint8_t>(),row_symbol_byte_offsets.data_ptr<std::int32_t>(),output.data_ptr(),layout,MXFP8,fp32,stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

void launch_dense_gemm_nt(const torch::Tensor& input,const torch::Tensor& weight,
        torch::Tensor& output,cudaStream_t stream) {
    auto handle=at::cuda::getCurrentCUDABlasHandle();
    TORCH_CHECK(cublasSetStream(handle,stream)==CUBLAS_STATUS_SUCCESS,"FP8-SQ cublasSetStream failed");
    const bool half=input.scalar_type()==at::kHalf;
    TORCH_CHECK(raw::gemm(handle,input.data_ptr(),weight.data_ptr(),output.data_ptr(),int(input.size(0)),int(input.size(1)),int(weight.size(0)),half?::tyloi::cuda::NumericType::f16 : ::tyloi::cuda::NumericType::f32,stream)==CUBLAS_STATUS_SUCCESS,
        half?"FP8-SQ FP16 GEMM failed":"FP8-SQ FP32 GEMM failed");
}
template <bool MXFP8>
torch::Tensor matmul(
        torch::Tensor blob,
        torch::Tensor row_q,
        torch::Tensor row_symbol_byte_offsets,
        torch::Tensor input,
        Layout layout) {
    validate_metadata(blob, row_q, row_symbol_byte_offsets, layout, MXFP8);
    TORCH_CHECK(
        input.is_cuda() && input.get_device() == blob.get_device() &&
        input.dim() == 2 && input.is_contiguous() &&
        input.size(1) == layout.width &&
        input.size(0) <= std::numeric_limits<int>::max() &&
        (input.scalar_type() == at::kHalf ||
         input.scalar_type() == at::kFloat),
        "FP8-SQ activation must be contiguous rank-2 CUDA FP16/FP32");
    const c10::cuda::CUDAGuard guard(blob.device());
    torch::Tensor output = torch::empty(
        {input.size(0), layout.outputs}, input.options());
    const int rows = static_cast<int>(input.size(0));
    if (rows == 0) return output;
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (rows > kDirectPackedMaxRows) {
        // The decoded matrix is a transient workspace owned by this call.  It
        // is never cached or attached to the packed weight.
        auto weight = dequant<MXFP8>(
            blob, row_q, row_symbol_byte_offsets, layout,
            input.scalar_type() == at::kFloat);
        launch_dense_gemm_nt(input, weight, output, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
    }
    raw::matmul(blob.data_ptr<std::uint8_t>(),row_q.data_ptr<std::uint8_t>(),row_symbol_byte_offsets.data_ptr<std::int32_t>(),input.data_ptr(),output.data_ptr(),layout,rows,MXFP8,input.scalar_type()==at::kFloat ? ::tyloi::cuda::NumericType::f32 : ::tyloi::cuda::NumericType::f16,stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template <bool MXFP8>
void routed_matmul(
        torch::Tensor blob,
        torch::Tensor row_q,
        torch::Tensor row_symbol_byte_offsets,
        torch::Tensor input,
        torch::Tensor expert_ids,
        torch::Tensor expert_local,
        std::int64_t n_experts,
        std::int64_t local_experts,
        std::int64_t out_per_expert,
        Layout layout,
        torch::Tensor output) {
    validate_metadata(blob, row_q, row_symbol_byte_offsets, layout, MXFP8);
    TORCH_CHECK(
        layout.outputs == local_experts * out_per_expert &&
        input.is_cuda() && input.get_device() == blob.get_device() &&
        input.is_contiguous() && input.scalar_type() == at::kHalf &&
        (input.dim() == 2 || input.dim() == 3) && input.size(-1) == layout.width &&
        expert_ids.is_cuda() && expert_ids.get_device() == blob.get_device() &&
        expert_ids.is_contiguous() &&
        expert_ids.scalar_type() == at::kInt &&
        expert_ids.dim() == 2 && expert_ids.size(0) == input.size(0) &&
        expert_local.is_cuda() && expert_local.get_device() == blob.get_device() &&
        expert_local.is_contiguous() &&
        expert_local.scalar_type() == at::kInt &&
        expert_local.dim() == 1 && expert_local.numel() == n_experts &&
        output.is_cuda() && output.get_device() == blob.get_device() &&
        output.is_contiguous() && output.scalar_type() == at::kHalf &&
        output.sizes() == c10::IntArrayRef(
            {expert_ids.size(0), expert_ids.size(1), out_per_expert}) &&
        (input.dim() == 2 || input.size(1) == expert_ids.size(1)) &&
        n_experts > 0 && n_experts <= std::numeric_limits<int>::max() &&
        local_experts > 0 && local_experts <= std::numeric_limits<int>::max() &&
        out_per_expert > 0 && out_per_expert <= std::numeric_limits<int>::max(),
        "FP8-SQ routed matmul requires matching contiguous CUDA FP16 tensors");
    TORCH_CHECK(
        expert_ids.numel() <= std::numeric_limits<int>::max(),
        "FP8-SQ route count exceeds CUDA limits");
    if (expert_ids.numel() == 0) return;
    const c10::cuda::CUDAGuard guard(blob.device());
    raw::routed(blob.data_ptr<std::uint8_t>(),row_q.data_ptr<std::uint8_t>(),row_symbol_byte_offsets.data_ptr<std::int32_t>(),input.data_ptr(),output.data_ptr(),expert_ids.data_ptr<std::int32_t>(),expert_local.data_ptr<std::int32_t>(),layout,int(expert_ids.numel()),int(expert_ids.size(1)),int(n_experts),int(local_experts),int(out_per_expert),input.dim()==2,MXFP8,at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}
torch::Tensor mxfp8_sq_dequant_cuda(
        torch::Tensor blob,
        torch::Tensor row_q,
        torch::Tensor row_symbol_byte_offsets,
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

torch::Tensor fp8_128_sq_dequant_cuda(
        torch::Tensor blob,
        torch::Tensor row_q,
        torch::Tensor row_symbol_byte_offsets,
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

torch::Tensor mxfp8_sq_matmul_cuda(
        torch::Tensor blob,
        torch::Tensor row_q,
        torch::Tensor row_symbol_byte_offsets,
        torch::Tensor input,
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

torch::Tensor fp8_128_sq_matmul_cuda(
        torch::Tensor blob,
        torch::Tensor row_q,
        torch::Tensor row_symbol_byte_offsets,
        torch::Tensor input,
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
        torch::Tensor blob,
        torch::Tensor row_q,
        torch::Tensor row_symbol_byte_offsets,
        torch::Tensor input,
        torch::Tensor expert_ids,
        torch::Tensor expert_local,
        std::int64_t n_experts, std::int64_t local_experts,
        std::int64_t out_per_expert, std::int64_t width,
        std::int64_t block_rows, std::int64_t block_columns,
        std::int64_t scale_rows, std::int64_t scale_columns,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset,
        torch::Tensor output) {
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
        torch::Tensor blob,
        torch::Tensor row_q,
        torch::Tensor row_symbol_byte_offsets,
        torch::Tensor input,
        torch::Tensor expert_ids,
        torch::Tensor expert_local,
        std::int64_t n_experts, std::int64_t local_experts,
        std::int64_t out_per_expert, std::int64_t width,
        std::int64_t scale_kind,
        std::int64_t palettes_offset, std::int64_t symbols_offset,
        std::int64_t scales_offset,
        torch::Tensor output) {
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
