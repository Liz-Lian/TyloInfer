#include "mxfp4_sq.h"
#include "mxfp4_sq_check.h"
#include "mxfp4_sq_backward_raw.h"
#include "packed_backward.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <limits>

using mfq_jit_sq4::validate;

torch::Tensor mxfp4_sq_backward_input_cuda(
        torch::Tensor blob,
        torch::Tensor row_q,
        torch::Tensor row_symbol_byte_offsets,
        torch::Tensor row_auxiliary,
        torch::Tensor output_gradient,
        std::int64_t bits,
        std::int64_t outputs,
        std::int64_t width,
        std::int64_t base,
        std::int64_t q_sum,
        std::int64_t sq4_rows) {
    const auto q = validate(
        blob, row_q, row_symbol_byte_offsets, row_auxiliary,
        bits, outputs, width, base, q_sum, sq4_rows);
    TORCH_CHECK(
        output_gradient.is_cuda() &&
        output_gradient.get_device() == blob.get_device() &&
        output_gradient.dim() == 2 && output_gradient.is_contiguous() &&
        output_gradient.size(1) == outputs &&
        (output_gradient.scalar_type() == at::kHalf ||
         output_gradient.scalar_type() == at::kFloat),
        "MXFP4-SQ backward requires contiguous CUDA FP16/FP32 [M,N]");
    const c10::cuda::CUDAGuard guard(blob.device());
    auto result = torch::empty(
        {output_gradient.size(0), width}, output_gradient.options());
    const auto count = output_gradient.size(0) * width;
    if (!count) return result;
    constexpr int threads = 256;
    const int blocks = int(std::min<std::int64_t>(
        (count + threads - 1) / threads, 65535));
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto* data = blob.data_ptr<std::uint8_t>();
    const int rows = int(output_gradient.size(0));
    if (output_gradient.scalar_type() == at::kHalf &&
            rows <= 8) {
        const int output_tile = rows <= 4 ? 16 : 32;
        const int splits = (int(outputs) + output_tile - 1) / output_tile;
        auto partials = torch::empty(
            {splits, rows, width},
            output_gradient.options().dtype(at::kFloat));
        const auto* gradient = reinterpret_cast<const __half*>(
            output_gradient.data_ptr<at::Half>());
        auto* destination = reinterpret_cast<__half*>(
            result.data_ptr<at::Half>());
        mfq_jit_sq4::backward_matrix(
            data,
            row_q.data_ptr<std::uint8_t>(),
            row_symbol_byte_offsets.data_ptr<std::int32_t>(),
            row_auxiliary.data_ptr<std::int32_t>(),
            gradient, partials.data_ptr<float>(), q, rows,
            output_tile, stream);
        mfq_jit_backward::launch_split_float_reduce_to_half(
            partials.data_ptr<float>(), destination,
            rows, int(width), splits, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return result;
    }
    auto weight = mxfp4_sq_dequant_cuda(
        blob, row_q, row_symbol_byte_offsets, row_auxiliary,
        bits, outputs, width, base, q_sum, sq4_rows, false);
    mfq_jit_backward::launch_dense_half_weight(
        output_gradient, weight, result,
        rows, int(outputs), int(width), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}