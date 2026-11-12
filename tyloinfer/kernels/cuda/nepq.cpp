#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "nepq_raw.h"

/** Apply the signed block Hadamard transform to CUDA activations. */
torch::Tensor nepq_hadamard_input_cuda(
    torch::Tensor input,
    torch::Tensor signs,
    int64_t block_size) {
    TORCH_CHECK(
        input.is_cuda() && input.scalar_type() == at::kHalf &&
        input.is_contiguous() && input.dim() == 2,
        "NEPQ Hadamard input must be CUDA contiguous fp16 rank-2");
    TORCH_CHECK(
        signs.is_cuda() && signs.scalar_type() == at::kChar &&
        signs.is_contiguous() && signs.dim() == 1,
        "NEPQ Hadamard signs must be CUDA contiguous int8 rank-1");
    TORCH_CHECK(
        block_size >= 2 && block_size <= 2048 &&
        (block_size & (block_size - 1)) == 0,
        "NEPQ Hadamard block must be a power of two in [2,2048]");
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    TORCH_CHECK(signs.numel() == width, "NEPQ Hadamard sign width mismatch");
    TORCH_CHECK(width % block_size == 0, "NEPQ Hadamard block must divide K");
    c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::empty_like(input);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    tyloi::cuda::nepq_raw::input(input.data_ptr(), signs.data_ptr<int8_t>(),
        output.data_ptr(), rows, width, static_cast<int>(block_size), stream);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "NEPQ Hadamard input kernel launch failed");
    return output;
}


/** Apply the adjoint signed block Hadamard transform to CUDA activations. */
torch::Tensor nepq_hadamard_adjoint_cuda(
    torch::Tensor input,
    torch::Tensor signs,
    int64_t block_size) {
    TORCH_CHECK(
        input.is_cuda() && input.scalar_type() == at::kHalf &&
        input.is_contiguous() && input.dim() == 2,
        "NEPQ Hadamard adjoint input must be CUDA contiguous fp16 rank-2");
    TORCH_CHECK(
        signs.is_cuda() && signs.scalar_type() == at::kChar &&
        signs.is_contiguous() && signs.dim() == 1,
        "NEPQ Hadamard adjoint signs must be CUDA contiguous int8 rank-1");
    TORCH_CHECK(
        block_size >= 2 && block_size <= 2048 &&
        (block_size & (block_size - 1)) == 0,
        "NEPQ Hadamard adjoint block must be a power of two in [2,2048]");
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    TORCH_CHECK(
        signs.numel() == width && width % block_size == 0,
        "NEPQ Hadamard adjoint geometry mismatch");
    TORCH_CHECK(
        input.device() == signs.device(),
        "NEPQ Hadamard adjoint tensors must share one CUDA device");
    c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::empty_like(input);
    if (rows == 0) {
        return output;
    }
    tyloi::cuda::nepq_raw::adjoint(input.data_ptr(), signs.data_ptr<int8_t>(),
        output.data_ptr(), rows, width, static_cast<int>(block_size),
        at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
