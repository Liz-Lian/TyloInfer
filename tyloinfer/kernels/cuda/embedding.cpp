#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "embedding_raw.h"
#include <cstdint>
#include <limits>

torch::Tensor embedding_lookup_cuda(
        torch::Tensor weight,
        torch::Tensor token_ids) {
    TORCH_CHECK(
        weight.is_cuda() && weight.is_contiguous() && weight.dim() == 2,
        "embedding weight must be contiguous CUDA rank-2");
    TORCH_CHECK(
        weight.scalar_type() == at::kFloat ||
        weight.scalar_type() == at::kHalf,
        "embedding weight must be fp16 or fp32");
    TORCH_CHECK(
        token_ids.is_cuda() && token_ids.is_contiguous() &&
        token_ids.scalar_type() == at::kLong &&
        token_ids.device() == weight.device(),
        "embedding token IDs must be contiguous CUDA int64 on the weight device");
    const int vocabulary = static_cast<int>(weight.size(0));
    const int width = static_cast<int>(weight.size(1));
    const int token_count = static_cast<int>(token_ids.numel());
    auto shape = token_ids.sizes().vec();
    shape.push_back(width);
    auto output = torch::empty(shape, weight.options());
    if (token_count == 0) {
        return output;
    }
    tyloi::cuda::embedding_raw::dense(weight.data_ptr(), token_ids.data_ptr<int64_t>(), output.data_ptr(),
        token_count, width, vocabulary, weight.scalar_type() == at::kHalf
            ? tyloi::cuda::NumericType::f16 : tyloi::cuda::NumericType::f32, at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


#if TYLOINFER_ENABLE_MFQ_COMPAT
torch::Tensor nint_embedding_cuda(
        torch::Tensor bitstream,
        torch::Tensor row_q_bits,
        torch::Tensor row_q_bit_offsets,
        torch::Tensor subgroup_scale,
        torch::Tensor subgroup_minimum,
        torch::Tensor neuron_scale,
        torch::Tensor neuron_minimum,
        torch::Tensor token_ids,
        int64_t width,
        int64_t group_size) {
    TORCH_CHECK(
        bitstream.is_cuda() && bitstream.is_contiguous() &&
        bitstream.scalar_type() == at::kByte &&
        bitstream.dim() == 1,
        "NINT embedding bitstream must be contiguous CUDA uint8 rank-1");
    TORCH_CHECK(
        row_q_bits.is_cuda() && row_q_bits.is_contiguous() &&
        row_q_bits.scalar_type() == at::kByte &&
        row_q_bits.dim() == 1 && row_q_bit_offsets.is_cuda() &&
        row_q_bit_offsets.is_contiguous() &&
        row_q_bit_offsets.scalar_type() == at::kLong &&
        row_q_bit_offsets.dim() == 1,
        "NINT embedding q metadata is invalid");
    TORCH_CHECK(
        subgroup_scale.is_cuda() && subgroup_scale.is_contiguous() &&
        subgroup_scale.scalar_type() == at::kByte &&
        subgroup_scale.dim() == 2 && subgroup_minimum.is_cuda() &&
        subgroup_minimum.is_contiguous() &&
        subgroup_minimum.scalar_type() == at::kByte &&
        subgroup_minimum.sizes() == subgroup_scale.sizes(),
        "NINT embedding subgroup metadata is invalid");
    TORCH_CHECK(
        neuron_scale.is_cuda() && neuron_scale.is_contiguous() &&
        neuron_scale.scalar_type() == at::kFloat &&
        neuron_minimum.is_cuda() && neuron_minimum.is_contiguous() &&
        neuron_minimum.scalar_type() == at::kFloat,
        "NINT embedding neuron metadata is invalid");
    TORCH_CHECK(
        token_ids.is_cuda() && token_ids.is_contiguous() &&
        token_ids.scalar_type() == at::kLong,
        "NINT embedding token IDs must be contiguous CUDA int64");
    const int vocabulary = static_cast<int>(subgroup_scale.size(0));
    const int groups = static_cast<int>(subgroup_scale.size(1));
    TORCH_CHECK(
        vocabulary > 0 && groups > 0 && group_size >= 4 && group_size <= 64 &&
        width > 0 && width <= static_cast<int64_t>(groups) * group_size,
        "NINT embedding dimensions are invalid");
    TORCH_CHECK(
        row_q_bits.numel() == vocabulary &&
        row_q_bit_offsets.numel() == vocabulary &&
        neuron_scale.numel() == vocabulary &&
        neuron_minimum.numel() == vocabulary,
        "NINT embedding row metadata shape mismatch");
    TORCH_CHECK(
        bitstream.device() == row_q_bits.device() &&
        bitstream.device() == row_q_bit_offsets.device() &&
        bitstream.device() == subgroup_scale.device() &&
        bitstream.device() == subgroup_minimum.device() &&
        bitstream.device() == neuron_scale.device() &&
        bitstream.device() == neuron_minimum.device() &&
        bitstream.device() == token_ids.device(),
        "NINT embedding tensors must share one CUDA device");
    const int token_count = static_cast<int>(token_ids.numel());
    auto shape = token_ids.sizes().vec();
    shape.push_back(width);
    auto output = torch::empty(
        shape, bitstream.options().dtype(at::kHalf));
    if (token_count == 0) {
        return output;
    }
    tyloi::cuda::embedding_raw::nint(bitstream.data_ptr<uint8_t>(), row_q_bits.data_ptr<uint8_t>(),
        row_q_bit_offsets.data_ptr<int64_t>(), subgroup_scale.data_ptr<uint8_t>(), subgroup_minimum.data_ptr<uint8_t>(),
        neuron_scale.data_ptr<float>(), neuron_minimum.data_ptr<float>(), token_ids.data_ptr<int64_t>(), output.data_ptr(),
        token_count, vocabulary, groups, int(group_size), int(width), at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


torch::Tensor nint8_zero_embedding_lookup_cuda(
        torch::Tensor quantized,
        torch::Tensor scale,
        torch::Tensor token_ids,
        int64_t width) {
    TORCH_CHECK(
        quantized.is_cuda() && quantized.is_contiguous() &&
        quantized.scalar_type() == at::kByte &&
        quantized.dim() == 3 && quantized.size(2) == 32,
        "NINT8-0 embedding q must be contiguous CUDA uint8 [V,G,32]");
    TORCH_CHECK(
        scale.is_cuda() && scale.is_contiguous() &&
        scale.scalar_type() == at::kHalf &&
        scale.dim() == 2 && scale.size(0) == quantized.size(0) &&
        scale.size(1) == quantized.size(1),
        "NINT8-0 embedding scale must be contiguous CUDA fp16 [V,G]");
    TORCH_CHECK(
        token_ids.is_cuda() && token_ids.is_contiguous() &&
        token_ids.scalar_type() == at::kLong &&
        token_ids.device() == quantized.device() &&
        scale.device() == quantized.device(),
        "NINT8-0 embedding tensors must share one CUDA device");
    TORCH_CHECK(
        width > 0 && width <= quantized.size(1) * 32,
        "NINT8-0 embedding width is invalid");
    const int vocabulary = static_cast<int>(quantized.size(0));
    const int groups = static_cast<int>(quantized.size(1));
    const int token_count = static_cast<int>(token_ids.numel());
    auto shape = token_ids.sizes().vec();
    shape.push_back(width);
    auto output = torch::empty(
        shape, quantized.options().dtype(at::kHalf));
    if (token_count == 0) {
        return output;
    }
    tyloi::cuda::embedding_raw::nint8(quantized.data_ptr<uint8_t>(), scale.data_ptr(), token_ids.data_ptr<int64_t>(),
        output.data_ptr(), token_count, vocabulary, groups, int(width), at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

#endif
