#include "nepq_residual_backward_raw.h"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

void nepq_validate_residual(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, int64_t, int64_t, int64_t);

/** Accumulate the sparse residual input gradient into the supplied tensor. */
torch::Tensor nepq_sparse_residual_backward_input_cuda(
    torch::Tensor dictionary,
    torch::Tensor first,
    torch::Tensor second,
    torch::Tensor output_gradient,
    int64_t position_bits,
    int64_t block_vectors,
    torch::Tensor input_gradient) {
    TORCH_CHECK(
        output_gradient.is_cuda() &&
        output_gradient.scalar_type() == at::kHalf &&
        output_gradient.is_contiguous() && output_gradient.dim() == 2 &&
        output_gradient.size(1) == first.size(0),
        "NEPQ-A backward output gradient must be CUDA fp16 [M,rows]");
    TORCH_CHECK(
        input_gradient.is_cuda() &&
        input_gradient.scalar_type() == at::kHalf &&
        input_gradient.is_contiguous() && input_gradient.dim() == 2 &&
        input_gradient.size(0) == output_gradient.size(0),
        "NEPQ-A backward input gradient must be CUDA fp16 [M,K]");
    nepq_validate_residual(
        dictionary, first, second, input_gradient.size(1),
        position_bits, block_vectors);
    TORCH_CHECK(
        dictionary.device() == output_gradient.device() &&
        output_gradient.device() == input_gradient.device(),
        "NEPQ-A backward tensors must share one CUDA device");
    c10::cuda::CUDAGuard guard(input_gradient.device());
    if (input_gradient.numel() == 0) {
        return input_gradient;
    }
    mfq_jit_nepq_residual::backward_input(
        reinterpret_cast<const __half *>(dictionary.data_ptr<at::Half>()),
        first.data_ptr<int16_t>(), second.data_ptr<int16_t>(),
        reinterpret_cast<const __half *>(output_gradient.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(input_gradient.data_ptr<at::Half>()),
        output_gradient.size(0), static_cast<int>(first.size(0)),
        static_cast<int>(input_gradient.size(1)), first.size(1),
        static_cast<int>(position_bits), static_cast<int>(block_vectors),
        at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return input_gradient;
}


/** Accumulate a transposed sparse residual gradient, optionally rotating it. */
torch::Tensor nepq_sparse_residual_backward_transpose_cuda(
    torch::Tensor dictionary,
    torch::Tensor transpose_offsets,
    torch::Tensor transpose_rows,
    torch::Tensor transpose_dictionary,
    torch::Tensor output_gradient,
    torch::Tensor rotation_signs,
    bool fuse_rotation,
    torch::Tensor input_gradient) {
    TORCH_CHECK(
        dictionary.is_cuda() &&
        dictionary.scalar_type() == at::kHalf &&
        dictionary.is_contiguous() && dictionary.dim() == 2 &&
        dictionary.size(0) == 1024 && dictionary.size(1) == 8,
        "NEPQ-A transpose dictionary must be CUDA fp16 [1024,8]");
    TORCH_CHECK(
        transpose_offsets.is_cuda() &&
        transpose_offsets.scalar_type() == at::kInt &&
        transpose_offsets.is_contiguous() && transpose_offsets.dim() == 1,
        "NEPQ-A transpose offsets must be CUDA contiguous int32");
    TORCH_CHECK(
        transpose_rows.is_cuda() &&
        transpose_rows.scalar_type() == at::kInt &&
        transpose_rows.is_contiguous() && transpose_rows.dim() == 1 &&
        transpose_dictionary.is_cuda() &&
        transpose_dictionary.scalar_type() == at::kShort &&
        transpose_dictionary.is_contiguous() &&
        transpose_dictionary.dim() == 1 &&
        transpose_dictionary.numel() == transpose_rows.numel(),
        "NEPQ-A transpose records must be matching int32/int16 vectors");
    TORCH_CHECK(
        output_gradient.is_cuda() &&
        output_gradient.scalar_type() == at::kHalf &&
        output_gradient.is_contiguous() && output_gradient.dim() == 2,
        "NEPQ-A transpose gradient must be CUDA contiguous fp16 [M,rows]");
    TORCH_CHECK(
        input_gradient.is_cuda() &&
        input_gradient.scalar_type() == at::kHalf &&
        input_gradient.is_contiguous() && input_gradient.dim() == 2 &&
        input_gradient.size(0) == output_gradient.size(0) &&
        input_gradient.size(1) % 8 == 0,
        "NEPQ-A transpose output must be CUDA contiguous fp16 [M,K]");
    const int64_t vectors = input_gradient.size(1) / 8;
    TORCH_CHECK(
        transpose_offsets.numel() == vectors + 1,
        "NEPQ-A transpose offset count mismatch");
    TORCH_CHECK(
        rotation_signs.is_cuda() &&
        rotation_signs.scalar_type() == at::kChar &&
        rotation_signs.is_contiguous() && rotation_signs.dim() == 1 &&
        (!fuse_rotation || rotation_signs.numel() == input_gradient.size(1)),
        "NEPQ-A fused rotation signs must match K");
    TORCH_CHECK(
        dictionary.device() == transpose_offsets.device() &&
        dictionary.device() == transpose_rows.device() &&
        dictionary.device() == transpose_dictionary.device() &&
        dictionary.device() == output_gradient.device() &&
        dictionary.device() == rotation_signs.device() &&
        dictionary.device() == input_gradient.device(),
        "NEPQ-A transpose tensors must share one CUDA device");
    if (input_gradient.numel() == 0) return input_gradient;

    c10::cuda::CUDAGuard guard(input_gradient.device());
    mfq_jit_nepq_residual::backward_transpose(
        reinterpret_cast<const __half *>(dictionary.data_ptr<at::Half>()),
        transpose_offsets.data_ptr<int32_t>(), transpose_rows.data_ptr<int32_t>(),
        transpose_dictionary.data_ptr<int16_t>(),
        reinterpret_cast<const __half *>(output_gradient.data_ptr<at::Half>()),
        rotation_signs.data_ptr<int8_t>(),
        reinterpret_cast<__half *>(input_gradient.data_ptr<at::Half>()),
        output_gradient.size(0), static_cast<int>(output_gradient.size(1)),
        static_cast<int>(input_gradient.size(1)), vectors, fuse_rotation,
        at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return input_gradient;
}
