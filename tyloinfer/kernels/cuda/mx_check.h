#pragma once
#include <torch/extension.h>
#include <limits>
namespace mfq_jit_mx {
/** Return the MXFP8 scale-column shift. */
inline int mxfp8_column_shift(const torch::Tensor& values,const torch::Tensor& scales) {
    return values.size(1)/scales.size(1)==128?7:5;
}
/** Return the MXFP8 scale-row shift. */
inline int mxfp8_row_shift(const torch::Tensor& values,const torch::Tensor& scales) {
    return scales.size(0)==values.size(0)?0:mxfp8_column_shift(values,scales);
}

/** Validate MXFP8 values and scale geometry. */
inline void validate_mxfp8(
        const torch::Tensor & values,
        const torch::Tensor & scales) {
    TORCH_CHECK(values.is_cuda() && scales.is_cuda(),
                "MXFP8 values and scales must be CUDA tensors");
    TORCH_CHECK(values.scalar_type() == at::kByte &&
                    scales.scalar_type() == at::kByte,
                "MXFP8 values and scales must be uint8");
    TORCH_CHECK(values.is_contiguous() && scales.is_contiguous(),
                "MXFP8 values and scales must be contiguous");
    TORCH_CHECK(values.dim() == 2 && scales.dim() == 2,
                "MXFP8 values and scales must be rank-2");
    TORCH_CHECK(values.size(0) > 0 && values.size(1) > 0 &&
                    values.size(0) <= std::numeric_limits<int>::max() &&
                    values.size(1) <= std::numeric_limits<int>::max() &&
                    values.size(1) % 32 == 0,
                "MXFP8 input width must be divisible by 32");
    const bool block128=values.size(1)%128==0 && scales.size(0)==(values.size(0)+127)/128 && scales.size(1)==values.size(1)/128;
    const bool block32=scales.size(0)==(values.size(0)+31)/32 && scales.size(1)==values.size(1)/32;
    const bool row32=scales.size(0)==values.size(0) && scales.size(1)==values.size(1)/32;
    TORCH_CHECK(block128 || block32 || row32,"MXFP8 scale geometry mismatch");
}

/** Validate dense MXFP4 values and scale geometry. */
inline void validate_mxfp4_dense(
        const torch::Tensor & values,
        const torch::Tensor & scales) {
    TORCH_CHECK(values.is_cuda() && scales.is_cuda(),
                "MXFP4 values and scales must be CUDA tensors");
    TORCH_CHECK(values.scalar_type() == at::kByte &&
                    scales.scalar_type() == at::kByte,
                "MXFP4 values and scales must be uint8");
    TORCH_CHECK(values.is_contiguous() && scales.is_contiguous(),
                "MXFP4 values and scales must be contiguous");
    TORCH_CHECK(values.dim() == 2 && scales.dim() == 2 &&
                    values.size(0) > 0 && values.size(1) > 0 &&
                    values.size(0) <= std::numeric_limits<int>::max() &&
                    values.size(1) <= std::numeric_limits<int>::max() / 2 &&
                    values.size(1) % 16 == 0 &&
                    scales.size(0) == values.size(0) &&
                    scales.size(1) == values.size(1) / 16,
                "MXFP4 weight geometry mismatch");
}

/** Validate the routed MXFP4 tensor geometry. */
inline void validate_mxfp4_moe(
        const torch::Tensor & values,
        const torch::Tensor & scales,
        const torch::Tensor & input,
        const torch::Tensor & ids,
        const torch::Tensor & expert_local,
        const torch::Tensor & output,
        int64_t global_experts,
        int64_t pool_experts,
        int64_t out_per_expert,
        int64_t neuron_len) {
    TORCH_CHECK(values.is_cuda() && scales.is_cuda() && input.is_cuda() &&
                    ids.is_cuda() && expert_local.is_cuda() && output.is_cuda(),
                "MXFP4 routed tensors must be CUDA tensors");
    TORCH_CHECK(values.scalar_type() == at::kByte &&
                    scales.scalar_type() == at::kByte &&
                    input.scalar_type() == at::kHalf &&
                    ids.scalar_type() == at::kInt &&
                    expert_local.scalar_type() == at::kInt &&
                    output.scalar_type() == at::kHalf,
                "MXFP4 routed tensor dtypes are invalid");
    TORCH_CHECK(values.is_contiguous() && scales.is_contiguous() &&
                    input.is_contiguous() && ids.is_contiguous() &&
                    expert_local.is_contiguous() && output.is_contiguous(),
                "MXFP4 routed tensors must be contiguous");
    TORCH_CHECK(neuron_len > 0 && neuron_len % 32 == 0 &&
                    values.dim() == 2 &&
                    values.size(0) == pool_experts * out_per_expert &&
                    values.size(1) == neuron_len / 2 &&
                    scales.sizes() == c10::IntArrayRef(
                        {pool_experts * out_per_expert, neuron_len / 32}),
                "MXFP4 routed weight geometry mismatch");
    TORCH_CHECK((input.dim() == 2 || input.dim() == 3) &&
                    input.size(-1) == neuron_len && ids.dim() == 2 &&
                    input.size(0) == ids.size(0) &&
                    output.sizes() == c10::IntArrayRef(
                        {ids.size(0), ids.size(1), out_per_expert}) &&
                    expert_local.numel() == global_experts,
                "MXFP4 routed activation/route geometry mismatch");
}

} // namespace mfq_jit_mx
