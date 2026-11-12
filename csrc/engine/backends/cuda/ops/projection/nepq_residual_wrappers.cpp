#include "backend.h"
#include "nepq_residual_raw.h"

namespace {
void validate_residual(
    const ::tyloi::DenseTensor & dictionary,
    const ::tyloi::DenseTensor & first,
    const ::tyloi::DenseTensor & second,
    int64_t width,
    int64_t position_bits,
    int64_t block_vectors) {
    TYLOINFER_RUNTIME_CHECK(
        dictionary.is_cuda() && dictionary.scalar_type() == ::tyloi::kFloat16 &&
        dictionary.is_contiguous() && dictionary.dim() == 2 &&
        dictionary.size(0) == 1024 && dictionary.size(1) == 8,
        "NEPQ-A dictionary must be CUDA contiguous fp16 [1024,8]");
    TYLOINFER_RUNTIME_CHECK(
        first.is_cuda() && first.scalar_type() == ::tyloi::kInt16 &&
        first.is_contiguous() && first.dim() == 2,
        "NEPQ-A first records must be CUDA contiguous int16 rank-2");
    TYLOINFER_RUNTIME_CHECK(
        second.is_cuda() && second.scalar_type() == ::tyloi::kInt16 &&
        second.is_contiguous() && second.sizes() == first.sizes(),
        "NEPQ-A second records must be int16 and match the first-record layout");
    TYLOINFER_RUNTIME_CHECK(
        dictionary.device() == first.device() &&
        dictionary.device() == second.device(),
        "NEPQ-A residual tensors must share one CUDA device");
    TYLOINFER_RUNTIME_CHECK(width > 0 && width % 8 == 0, "NEPQ-A width must be divisible by 8");
    TYLOINFER_RUNTIME_CHECK(
        position_bits >= 1 && position_bits <= 5 &&
        block_vectors >= 2 && block_vectors <= 32,
        "invalid NEPQ-A residual profile");
    const int64_t expected_blocks = (width / 8 + block_vectors - 1) / block_vectors;
    TYLOINFER_RUNTIME_CHECK(first.size(1) == expected_blocks, "NEPQ-A block count mismatch");
}

}

/** Add a sparse residual matrix product to the supplied output. */
::tyloi::DenseTensor nepq_sparse_residual_matmul_cuda(
    ::tyloi::DenseTensor dictionary,
    ::tyloi::DenseTensor first,
    ::tyloi::DenseTensor second,
    ::tyloi::DenseTensor input,
    int64_t position_bits,
    int64_t block_vectors,
    ::tyloi::DenseTensor output) {
    TYLOINFER_RUNTIME_CHECK(
        input.is_cuda() && input.scalar_type() == ::tyloi::kFloat16 &&
        input.is_contiguous() && input.dim() == 2,
        "NEPQ-A input must be CUDA contiguous fp16 rank-2");
    TYLOINFER_RUNTIME_CHECK(
        output.is_cuda() && output.scalar_type() == ::tyloi::kFloat16 &&
        output.is_contiguous() && output.dim() == 2 &&
        output.size(0) == input.size(0) && output.size(1) == first.size(0),
        "NEPQ-A output must be CUDA contiguous fp16 [M,rows]");
    validate_residual(
        dictionary, first, second, input.size(1), position_bits, block_vectors);
    TYLOINFER_RUNTIME_CHECK(
        dictionary.device() == input.device() && input.device() == output.device(),
        "NEPQ-A matmul tensors must share one CUDA device");
    MfqCudaGuard guard(input.device());
    const int64_t total = input.size(0) * first.size(0);
    cudaStream_t stream = mfq_current_cuda_stream();
    tyloi::cuda::nepq_residual_raw::matmul(
        dictionary.data_ptr(), first.data_ptr<int16_t>(), second.data_ptr<int16_t>(),
        input.data_ptr(), output.data_ptr(), static_cast<int>(first.size(0)),
        static_cast<int>(input.size(1)), static_cast<int>(input.size(0)),
        static_cast<int>(first.size(1)), static_cast<int>(position_bits),
        static_cast<int>(block_vectors), total, stream);
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess,
                "NEPQ-A residual matmul kernel launch failed");
    return output;
}


/** Add decoded sparse residual vectors to the supplied weight matrix. */
::tyloi::DenseTensor nepq_sparse_residual_dequant_cuda(
    ::tyloi::DenseTensor dictionary,
    ::tyloi::DenseTensor first,
    ::tyloi::DenseTensor second,
    int64_t position_bits,
    int64_t block_vectors,
    ::tyloi::DenseTensor weight) {
    TYLOINFER_RUNTIME_CHECK(
        weight.is_cuda() && weight.scalar_type() == ::tyloi::kFloat16 &&
        weight.is_contiguous() && weight.dim() == 2 &&
        weight.size(0) == first.size(0),
        "NEPQ-A weight must be CUDA contiguous fp16 [rows,K]");
    validate_residual(
        dictionary, first, second, weight.size(1), position_bits, block_vectors);
    TYLOINFER_RUNTIME_CHECK(
        dictionary.device() == weight.device(),
        "NEPQ-A dequant tensors must share one CUDA device");
    MfqCudaGuard guard(weight.device());
    const int64_t total = first.numel();
    cudaStream_t stream = mfq_current_cuda_stream();
    tyloi::cuda::nepq_residual_raw::dequant(
        dictionary.data_ptr(), first.data_ptr<int16_t>(), second.data_ptr<int16_t>(),
        weight.data_ptr(), static_cast<int>(weight.size(1)),
        static_cast<int>(first.size(1)), static_cast<int>(position_bits),
        static_cast<int>(block_vectors), total, stream);
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess,
                "NEPQ-A residual dequant kernel launch failed");
    return weight;
}


/** Add routed sparse residual products to the supplied output. */
::tyloi::DenseTensor nepq_sparse_residual_grouped_cuda(
    ::tyloi::DenseTensor dictionary,
    ::tyloi::DenseTensor first,
    ::tyloi::DenseTensor second,
    ::tyloi::DenseTensor input,
    ::tyloi::DenseTensor route_ids,
    ::tyloi::DenseTensor expert_local,
    int64_t out_per_expert,
    int64_t position_bits,
    int64_t block_vectors,
    ::tyloi::DenseTensor output) {
    TYLOINFER_RUNTIME_CHECK(
        input.is_cuda() && input.scalar_type() == ::tyloi::kFloat16 &&
        input.is_contiguous() && (input.dim() == 2 || input.dim() == 3),
        "NEPQ-A routed input must be CUDA contiguous fp16 rank-2 or rank-3");
    TYLOINFER_RUNTIME_CHECK(
        route_ids.is_cuda() && route_ids.scalar_type() == ::tyloi::kInt32 &&
        route_ids.is_contiguous() && route_ids.dim() == 2,
        "NEPQ-A route IDs must be CUDA contiguous int32 rank-2");
    TYLOINFER_RUNTIME_CHECK(
        expert_local.is_cuda() && expert_local.scalar_type() == ::tyloi::kInt32 &&
        expert_local.is_contiguous() && expert_local.dim() == 1,
        "NEPQ-A expert map must be CUDA contiguous int32 rank-1");
    TYLOINFER_RUNTIME_CHECK(
        output.is_cuda() && output.scalar_type() == ::tyloi::kFloat16 &&
        output.is_contiguous() && output.dim() == 3 &&
        output.size(0) == route_ids.size(0) &&
        output.size(1) == route_ids.size(1) &&
        output.size(2) == out_per_expert,
        "NEPQ-A grouped output shape mismatch");
    const int64_t width = input.size(-1);
    validate_residual(dictionary, first, second, width, position_bits, block_vectors);
    TYLOINFER_RUNTIME_CHECK(
        dictionary.device() == input.device() &&
        input.device() == route_ids.device() &&
        input.device() == expert_local.device() &&
        input.device() == output.device(),
        "NEPQ-A grouped tensors must share one CUDA device");
    TYLOINFER_RUNTIME_CHECK(
        first.size(0) % out_per_expert == 0,
        "NEPQ-A expert row count is not divisible by output width");
    TYLOINFER_RUNTIME_CHECK(
        input.size(0) == route_ids.size(0) &&
        (input.dim() == 2 || input.size(1) == route_ids.size(1)),
        "NEPQ-A routed input leading dimensions mismatch");
    MfqCudaGuard guard(input.device());
    const int64_t total = route_ids.numel() * out_per_expert;
    cudaStream_t stream = mfq_current_cuda_stream();
    tyloi::cuda::nepq_residual_raw::grouped(
        dictionary.data_ptr(), first.data_ptr<int16_t>(), second.data_ptr<int16_t>(),
        input.data_ptr(), route_ids.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(),
        output.data_ptr(), static_cast<int>(route_ids.size(0)),
        static_cast<int>(route_ids.size(1)), static_cast<int>(out_per_expert),
        static_cast<int>(width), static_cast<int>(first.size(1)),
        static_cast<int>(position_bits), static_cast<int>(block_vectors),
        input.dim() == 3, expert_local.numel() != 0,
        static_cast<int>(expert_local.numel()),
        static_cast<int>(first.size(0) / out_per_expert), total, stream);
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess,
                "NEPQ-A residual grouped kernel launch failed");
    return output;
}
