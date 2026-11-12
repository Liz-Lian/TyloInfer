#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cstdint>
#include "mx_check.h"
#include "mx_backward_raw.h"
#include "packed_backward.h"

using mfq_jit_mx::mxfp8_column_shift;
using mfq_jit_mx::mxfp8_row_shift;
using mfq_jit_mx::validate_mxfp4_dense;
using mfq_jit_mx::validate_mxfp8;
torch::Tensor mxfp4_dequant_cuda(torch::Tensor,torch::Tensor);
torch::Tensor mxfp8_dequant_cuda(torch::Tensor,torch::Tensor);

/** Compute MX packed input gradients with the original short-row dispatch. */
torch::Tensor mx_backward_input_cuda(
        torch::Tensor values,
        torch::Tensor scales,
        torch::Tensor output_gradient,
        bool mxfp4) {
    if (mxfp4) {
        validate_mxfp4_dense(values, scales);
    } else {
        validate_mxfp8(values, scales);
    }
    TORCH_CHECK(
        output_gradient.is_cuda() && output_gradient.is_contiguous() &&
        output_gradient.dim() == 2 && output_gradient.size(1) == values.size(0),
        "MX backward output gradient geometry mismatch");
    TORCH_CHECK(
        values.device() == scales.device() &&
        values.device() == output_gradient.device(),
        "MX backward tensors must share one CUDA device");
    const auto dtype = output_gradient.scalar_type();
    TORCH_CHECK(
        dtype == at::kHalf ||
        dtype == at::kFloat ||
        dtype == at::kBFloat16,
        "MX backward output gradient must be fp16, bf16, or fp32");
    const int rows = static_cast<int>(output_gradient.size(0));
    const int outputs = static_cast<int>(values.size(0));
    const int width = static_cast<int>(mxfp4 ? values.size(1) * 2 : values.size(1));
    auto result = torch::empty({rows, width}, output_gradient.options());
    const int64_t total = static_cast<int64_t>(rows) * width;
    if (total == 0) {
        return result;
    }
    c10::cuda::CUDAGuard guard(output_gradient.device());
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<int64_t>(
        (total + threads - 1) / threads, 65535));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    using mfq_jit_mx::Backward;
    const bool small4=mxfp4 && dtype==at::kHalf && rows<=8;
    const bool small8=!mxfp4 && mxfp8_column_shift(values,scales)==7 && mxfp8_row_shift(values,scales)==7 && dtype==at::kHalf && rows<=8;
    if(small4 || small8) {
        auto kind=Backward::fp4;
        if(small8) {
            const int groups=(outputs+127)/128;
            if(rows==1 && groups<=8)kind=Backward::block;
            else if(rows<=2)kind=Backward::scalar;
            else if(rows<=4)kind=Backward::vec4;
            else if(static_cast<int64_t>(width)>=2*static_cast<int64_t>(outputs))kind=Backward::mma16;
            else if(2*static_cast<int64_t>(width)>=static_cast<int64_t>(outputs))kind=Backward::mma8;
            else kind=Backward::vec4;
        }
        torch::Tensor partials;
        if(kind==Backward::fp4 || kind==Backward::scalar || kind==Backward::vec4) {
            const int groups=kind==Backward::scalar?(outputs+127)/128:(outputs+31)/32;
            partials=torch::empty({groups,rows,width},values.options().dtype(at::kFloat));
        }
        mfq_jit_mx::backward(kind,values.data_ptr<uint8_t>(),scales.data_ptr<uint8_t>(),
            output_gradient.data_ptr<at::Half>(),result.data_ptr<at::Half>(),
            partials.defined()?partials.data_ptr<float>():nullptr,rows,outputs,width,stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return result;
    }
    auto weight = mxfp4
        ? mxfp4_dequant_cuda(values, scales)
        : mxfp8_dequant_cuda(values, scales);
    mfq_jit_backward::launch_dense_half_weight(
        output_gradient, weight, result, rows, outputs, width, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
