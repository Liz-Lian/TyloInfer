#include "backend.h"
#include "nepq_raw.h"

#include <algorithm>
#include <limits>

/** Apply the signed block Hadamard transform to CUDA activations. */
::tyloi::DenseTensor nepq_hadamard_input_cuda(
    ::tyloi::DenseTensor input,
    ::tyloi::DenseTensor signs,
    int64_t block_size) {
    TYLOINFER_RUNTIME_CHECK(
        input.is_cuda() && input.scalar_type() == ::tyloi::kFloat16 &&
        input.is_contiguous() && input.dim() == 2,
        "NEPQ Hadamard input must be CUDA contiguous fp16 rank-2");
    TYLOINFER_RUNTIME_CHECK(
        signs.is_cuda() && signs.scalar_type() == ::tyloi::kInt8 &&
        signs.is_contiguous() && signs.dim() == 1,
        "NEPQ Hadamard signs must be CUDA contiguous int8 rank-1");
    TYLOINFER_RUNTIME_CHECK(
        block_size >= 2 && block_size <= 2048 &&
        (block_size & (block_size - 1)) == 0,
        "NEPQ Hadamard block must be a power of two in [2,2048]");
    const int64_t rows = input.size(0);
    TYLOINFER_RUNTIME_CHECK(input.size(1) <= std::numeric_limits<int>::max(),
        "NEPQ Hadamard input width exceeds kernel limit");
    const int width = static_cast<int>(input.size(1));
    TYLOINFER_RUNTIME_CHECK(signs.numel() == width, "NEPQ Hadamard sign width mismatch");
    TYLOINFER_RUNTIME_CHECK(width % block_size == 0, "NEPQ Hadamard block must divide K");
    MfqCudaGuard guard(input.device());
    auto output = ::tyloi::cuda::empty_like(input);
    cudaStream_t stream = mfq_current_cuda_stream();
    const auto* source = static_cast<const __half*>(input.data_ptr());
    auto* destination = static_cast<__half*>(output.data_ptr());
    for (int64_t row = 0; row < rows;) {
        const int count = static_cast<int>(std::min<int64_t>(rows - row, 65535));
        const auto offset = row * static_cast<int64_t>(width);
        tyloi::cuda::nepq_raw::input(source + offset, signs.data_ptr<int8_t>(),
            destination + offset, count, width, static_cast<int>(block_size), stream);
        TYLOINFER_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess,
            "NEPQ Hadamard input kernel launch failed");
        row += count;
    }
    return output;
}


/** Apply the adjoint signed block Hadamard transform to CUDA activations. */
::tyloi::DenseTensor nepq_hadamard_adjoint_cuda(
    ::tyloi::DenseTensor input,
    ::tyloi::DenseTensor signs,
    int64_t block_size) {
    TYLOINFER_RUNTIME_CHECK(
        input.is_cuda() && input.scalar_type() == ::tyloi::kFloat16 &&
        input.is_contiguous() && input.dim() == 2,
        "NEPQ Hadamard adjoint input must be CUDA contiguous fp16 rank-2");
    TYLOINFER_RUNTIME_CHECK(
        signs.is_cuda() && signs.scalar_type() == ::tyloi::kInt8 &&
        signs.is_contiguous() && signs.dim() == 1,
        "NEPQ Hadamard adjoint signs must be CUDA contiguous int8 rank-1");
    TYLOINFER_RUNTIME_CHECK(
        block_size >= 2 && block_size <= 2048 &&
        (block_size & (block_size - 1)) == 0,
        "NEPQ Hadamard adjoint block must be a power of two in [2,2048]");
    const int64_t rows = input.size(0);
    TYLOINFER_RUNTIME_CHECK(input.size(1) <= std::numeric_limits<int>::max(),
        "NEPQ Hadamard adjoint width exceeds kernel limit");
    const int width = static_cast<int>(input.size(1));
    TYLOINFER_RUNTIME_CHECK(
        signs.numel() == width && width % block_size == 0,
        "NEPQ Hadamard adjoint geometry mismatch");
    TYLOINFER_RUNTIME_CHECK(
        input.device() == signs.device(),
        "NEPQ Hadamard adjoint tensors must share one CUDA device");
    MfqCudaGuard guard(input.device());
    auto output = ::tyloi::cuda::empty_like(input);
    const auto* source = static_cast<const __half*>(input.data_ptr());
    auto* destination = static_cast<__half*>(output.data_ptr());
    const auto stream = mfq_current_cuda_stream();
    for (int64_t row = 0; row < rows;) {
        const int count = static_cast<int>(std::min<int64_t>(rows - row, 65535));
        const auto offset = row * static_cast<int64_t>(width);
        tyloi::cuda::nepq_raw::adjoint(source + offset, signs.data_ptr<int8_t>(),
            destination + offset, count, width, static_cast<int>(block_size), stream);
        TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
        row += count;
    }
    return output;
}
