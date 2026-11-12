#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "activation_raw.h"

namespace {
using Type = tyloi::cuda::activation_raw::Type;
Type raw_type(at::ScalarType type) {
    if (type == at::kFloat) return Type::f32;
    if (type == at::kHalf) return Type::f16;
    if (type == at::kBFloat16) return Type::bf16;
    TORCH_CHECK(false, "unsupported activation dtype");
}
}

torch::Tensor silu_mul_cuda(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.is_cuda() && gate.is_contiguous(), "silu_mul: gate must be cuda contiguous");
    TORCH_CHECK(up.is_cuda() && up.is_contiguous(), "silu_mul: up must be cuda contiguous");
    TORCH_CHECK(gate.sizes() == up.sizes(), "silu_mul: gate/up shapes must match");
    TORCH_CHECK(gate.scalar_type() == up.scalar_type(), "silu_mul: gate/up dtype must match");
    TORCH_CHECK(gate.scalar_type() == at::kFloat || gate.scalar_type() == at::kHalf || gate.scalar_type() == at::kBFloat16,
        "silu_mul: dtype must be f32, f16, or bf16");
    auto out = torch::empty_like(gate);
    tyloi::cuda::activation_raw::silu_mul(gate.data_ptr(), up.data_ptr(), out.data_ptr(),
        gate.numel(), raw_type(gate.scalar_type()), at::cuda::getCurrentCUDAStream().stream());
    return out;
}

torch::Tensor gelu_mul_cuda(torch::Tensor gate, torch::Tensor up) {
    TORCH_CHECK(gate.is_cuda() && gate.is_contiguous(), "gelu_mul: gate must be cuda contiguous");
    TORCH_CHECK(up.is_cuda() && up.is_contiguous(), "gelu_mul: up must be cuda contiguous");
    TORCH_CHECK(gate.sizes() == up.sizes(), "gelu_mul: gate/up shapes must match");
    TORCH_CHECK(gate.scalar_type() == up.scalar_type(), "gelu_mul: gate/up dtype must match");
    TORCH_CHECK(gate.scalar_type() == at::kFloat || gate.scalar_type() == at::kHalf,
        "gelu_mul: dtype must be f32 or f16");
    auto out = torch::empty_like(gate);
    tyloi::cuda::activation_raw::gelu_mul(gate.data_ptr(), up.data_ptr(), out.data_ptr(),
        gate.numel(), raw_type(gate.scalar_type()), at::cuda::getCurrentCUDAStream().stream());
    return out;
}
