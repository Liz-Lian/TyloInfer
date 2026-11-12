#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "norm_raw.h"
#include <stdexcept>

namespace {
tyloi::cuda::NumericType raw_type(at::ScalarType type) {
    if (type == at::kFloat) return tyloi::cuda::NumericType::f32;
    if (type == at::kHalf) return tyloi::cuda::NumericType::f16;
    throw std::invalid_argument("unsupported norm dtype");
}
}

torch::Tensor rms_norm_cuda(torch::Tensor x, torch::Tensor weight, double eps)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kFloat,
                "rms_norm: x must be cuda contiguous f32");
    TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == at::kFloat, "weight must be cuda f32");
    int D = (int)x.size(-1);
    int N = (int)(x.numel() / D);
    c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty_like(x);
    tyloi::cuda::norm_raw::rms(x.data_ptr(), weight.data_ptr<float>(), out.data_ptr(), N, D, float(eps), 0.0f, raw_type(x.scalar_type()), at::cuda::getCurrentCUDAStream().stream());
    return out;
}

torch::Tensor rms_norm_offset_cuda(torch::Tensor x, torch::Tensor weight, double eps, double weight_offset)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kFloat,
                "rms_norm_offset: x must be cuda contiguous f32");
    TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == at::kFloat, "weight must be cuda f32");
    int D = (int)x.size(-1);
    int N = (int)(x.numel() / D);
    c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty_like(x);
    tyloi::cuda::norm_raw::rms(x.data_ptr(), weight.data_ptr<float>(), out.data_ptr(), N, D, float(eps), float(weight_offset), raw_type(x.scalar_type()), at::cuda::getCurrentCUDAStream().stream());
    return out;
}

torch::Tensor rms_norm_f16_cuda(torch::Tensor x, torch::Tensor weight, double eps,
                                double weight_offset)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kHalf,
                "rms_norm_f16: x must be cuda contiguous f16");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() &&
                    weight.scalar_type() == at::kFloat,
                "rms_norm_f16: weight must be cuda contiguous f32");
    int D = (int)x.size(-1);
    int N = (int)(x.numel() / D);
    TORCH_CHECK(weight.numel() == D, "rms_norm_f16: weight length mismatch");
    c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty_like(x);
    tyloi::cuda::norm_raw::rms(x.data_ptr(), weight.data_ptr<float>(), out.data_ptr(), N, D, float(eps), float(weight_offset), raw_type(x.scalar_type()), at::cuda::getCurrentCUDAStream().stream());
    return out;
}

torch::Tensor l2_norm_cuda(torch::Tensor x, double eps)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kFloat,
                "l2_norm: x must be cuda contiguous f32");
    int D = (int)x.size(-1);
    int N = (int)(x.numel() / D);
    c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty_like(x);
    tyloi::cuda::norm_raw::l2(x.data_ptr<float>(), out.data_ptr<float>(), N, D, float(eps), at::cuda::getCurrentCUDAStream().stream());
    return out;
}
