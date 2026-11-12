#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "nint_backward_raw.h"
#include "packed_backward.h"
#include "nint_check.h"

/** Decode NINT8-zero weights to FP16. */
torch::Tensor nint8_zero_dequant_cuda(torch::Tensor, torch::Tensor, int64_t);

/** Compute packed NINT input gradients for small FP16 batches. */
torch::Tensor nint_backward_input_cuda(
        torch::Tensor bitstream,
        torch::Tensor row_q_bits,
        torch::Tensor row_q_bit_offsets,
        torch::Tensor subgroup_scale,
        torch::Tensor subgroup_minimum,
        torch::Tensor neuron_scale,
        torch::Tensor neuron_minimum,
        torch::Tensor output_gradient,
        int64_t width,
        int64_t group_size) {
    TORCH_CHECK(
        output_gradient.is_cuda() && output_gradient.is_contiguous() &&
        output_gradient.scalar_type() == at::kHalf &&
        output_gradient.dim() == 2,
        "NINT output gradient must be contiguous CUDA fp16 rank-2");
    TORCH_CHECK(
        bitstream.is_cuda() && bitstream.is_contiguous() &&
        bitstream.scalar_type() == at::kByte &&
        bitstream.dim() == 1 && row_q_bits.is_cuda() &&
        row_q_bits.is_contiguous() &&
        row_q_bits.scalar_type() == at::kByte &&
        row_q_bits.dim() == 1 && row_q_bit_offsets.is_cuda() &&
        row_q_bit_offsets.is_contiguous() &&
        row_q_bit_offsets.scalar_type() == at::kLong &&
        row_q_bit_offsets.dim() == 1,
        "NINT bitstream or q metadata is invalid");
    TORCH_CHECK(
        subgroup_scale.is_cuda() && subgroup_scale.is_contiguous() &&
        subgroup_scale.scalar_type() == at::kByte &&
        subgroup_scale.dim() == 2 && subgroup_minimum.is_cuda() &&
        subgroup_minimum.is_contiguous() &&
        subgroup_minimum.scalar_type() == at::kByte &&
        subgroup_minimum.sizes() == subgroup_scale.sizes(),
        "NINT subgroup metadata is invalid");
    TORCH_CHECK(
        neuron_scale.is_cuda() && neuron_scale.is_contiguous() &&
        neuron_scale.scalar_type() == at::kFloat &&
        neuron_minimum.is_cuda() && neuron_minimum.is_contiguous() &&
        neuron_minimum.scalar_type() == at::kFloat,
        "NINT neuron metadata is invalid");
    const int output_rows = static_cast<int>(subgroup_scale.size(0));
    const int groups = static_cast<int>(subgroup_scale.size(1));
    const int activation_rows = static_cast<int>(output_gradient.size(0));
    TORCH_CHECK(
        activation_rows >= 1 && activation_rows <= 8 &&
        output_gradient.size(1) == output_rows && group_size >= 4 &&
        group_size <= 64 && width > 0 &&
        width <= static_cast<int64_t>(groups) * group_size,
        "NINT packed backward dimensions are invalid");
    TORCH_CHECK(
        row_q_bits.numel() == output_rows &&
        row_q_bit_offsets.numel() == output_rows &&
        neuron_scale.numel() == output_rows &&
        neuron_minimum.numel() == output_rows,
        "NINT backward row metadata shape mismatch");
    TORCH_CHECK(
        bitstream.device() == output_gradient.device() &&
        row_q_bits.device() == output_gradient.device() &&
        row_q_bit_offsets.device() == output_gradient.device() &&
        subgroup_scale.device() == output_gradient.device() &&
        subgroup_minimum.device() == output_gradient.device() &&
        neuron_scale.device() == output_gradient.device() &&
        neuron_minimum.device() == output_gradient.device(),
        "NINT backward tensors must share one CUDA device");
    auto result = torch::empty(
        {activation_rows, width}, output_gradient.options());
    mfq_jit_nint::backward(bitstream.data_ptr<uint8_t>(),row_q_bits.data_ptr<uint8_t>(),
        row_q_bit_offsets.data_ptr<int64_t>(),subgroup_scale.data_ptr<uint8_t>(),subgroup_minimum.data_ptr<uint8_t>(),
        neuron_scale.data_ptr<float>(),neuron_minimum.data_ptr<float>(),
        output_gradient.data_ptr<at::Half>(),result.data_ptr<at::Half>(),activation_rows,output_rows,groups,
        static_cast<int>(width),static_cast<int>(group_size),at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}


/** Compute NINT8-zero input gradients using decoded FP16 weights. */
torch::Tensor nint8_zero_backward_input_cuda(
        torch::Tensor quantized,
        torch::Tensor scale,
        torch::Tensor output_gradient,
        int64_t width) {
    mfq_jit_nint::validate_nint8_zero(quantized, scale);
    TORCH_CHECK(
        output_gradient.is_cuda() && output_gradient.is_contiguous() &&
        output_gradient.dim() == 2 &&
        output_gradient.size(1) == quantized.size(0),
        "NINT8-0 output-gradient geometry is invalid");
    TORCH_CHECK(
        output_gradient.scalar_type() == at::kHalf ||
        output_gradient.scalar_type() == at::kBFloat16 ||
        output_gradient.scalar_type() == at::kFloat,
        "NINT8-0 output gradient must be fp16, bf16, or fp32");
    TORCH_CHECK(
        width > 0 && width <= quantized.size(1) * 32 &&
        quantized.device() == output_gradient.device(),
        "NINT8-0 backward dimensions or device are invalid");
    const int rows = static_cast<int>(output_gradient.size(0));
    const int outputs = static_cast<int>(quantized.size(0));
    auto weight = nint8_zero_dequant_cuda(quantized, scale, width);
    auto result = torch::empty(
        {rows, width}, output_gradient.options());
    mfq_jit_backward::launch_dense_half_weight(
        output_gradient,
        weight,
        result,
        rows,
        outputs,
        static_cast<int>(width),
        at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
