#include "backend.h"
#include "mfq_cuda_kernels.h"
#include "activation_raw.h"
#include <array>
#include <stdexcept>
#include <vector>

namespace {
tyloi::cuda::activation_raw::Type raw_type(tyloi::ScalarType type) {
    using Type = tyloi::cuda::activation_raw::Type;
    if (type == tyloi::kFloat32) return Type::f32;
    if (type == tyloi::kFloat16) return Type::f16;
    if (type == tyloi::kBFloat16) return Type::bf16;
    throw std::invalid_argument("unsupported activation dtype");
}
}

::tyloi::DenseTensor clipped_silu_mul_cuda(::tyloi::DenseTensor gate,
        ::tyloi::DenseTensor up,double limit) {
    TYLOINFER_RUNTIME_CHECK(gate.is_cuda() && up.device()==gate.device() &&
        gate.scalar_type()==::tyloi::kFloat16 && up.scalar_type()==::tyloi::kFloat16 &&
        gate.sizes()==up.sizes() && gate.is_contiguous() && up.is_contiguous() && limit>0,
        "clipped SwiGLU requires matching contiguous CUDA FP16 tensors and a positive limit");
    auto output=::tyloi::cuda::empty(gate.sizes(),gate.options());
    tyloi::cuda::activation_raw::clipped_silu_mul(gate.data_ptr(), up.data_ptr(), output.data_ptr(), gate.numel(), float(limit), mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

::tyloi::DenseTensor silu_mul_cuda(::tyloi::DenseTensor gate, ::tyloi::DenseTensor up)
{
    TYLOINFER_RUNTIME_CHECK(gate.is_cuda() && gate.is_contiguous(), "silu_mul: gate must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(up.is_cuda() && up.is_contiguous(), "silu_mul: up must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(gate.sizes() == up.sizes(), "silu_mul: gate/up shapes must match");
    TYLOINFER_RUNTIME_CHECK(gate.scalar_type() == up.scalar_type(), "silu_mul: gate/up dtype must match");
    TYLOINFER_RUNTIME_CHECK(
        gate.scalar_type() == ::tyloi::kFloat32 ||
        gate.scalar_type() == ::tyloi::kFloat16 ||
        gate.scalar_type() == ::tyloi::kBFloat16,
        "silu_mul: dtype must be f32, f16, or bf16");

    auto out = ::tyloi::cuda::empty_like(gate);
    const auto dtype = gate.scalar_type() == ::tyloi::kFloat32
        ? ::tyloi::ScalarType::float32
        : gate.scalar_type() == ::tyloi::kFloat16
            ? ::tyloi::ScalarType::float16
            : ::tyloi::ScalarType::bfloat16;
    const auto shape = gate.sizes().vec();
    tyloi::cuda::kernels::silu_mul(
        ::tyloi::make_contiguous_view(
            gate.data_ptr(), shape, dtype,
            ::tyloi::Device{::tyloi::DeviceType::cuda, gate.get_device()}),
        ::tyloi::make_contiguous_view(
            up.data_ptr(), shape, dtype,
            ::tyloi::Device{::tyloi::DeviceType::cuda, up.get_device()}),
        ::tyloi::make_contiguous_view(
            out.data_ptr(), shape, dtype,
            ::tyloi::Device{::tyloi::DeviceType::cuda, out.get_device()}),
        mfq_current_cuda_stream());
    return out;
}

::tyloi::DenseTensor gelu_mul_cuda(::tyloi::DenseTensor gate, ::tyloi::DenseTensor up)
{
    TYLOINFER_RUNTIME_CHECK(gate.is_cuda() && gate.is_contiguous(), "gelu_mul: gate must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(up.is_cuda() && up.is_contiguous(), "gelu_mul: up must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(gate.sizes() == up.sizes(), "gelu_mul: gate/up shapes must match");
    TYLOINFER_RUNTIME_CHECK(gate.scalar_type() == up.scalar_type(), "gelu_mul: gate/up dtype must match");
    TYLOINFER_RUNTIME_CHECK(gate.scalar_type() == ::tyloi::kFloat32 || gate.scalar_type() == ::tyloi::kFloat16,
                "gelu_mul: dtype must be f32 or f16");

    auto out = ::tyloi::cuda::empty_like(gate);
    const auto dtype = gate.scalar_type() == ::tyloi::kFloat32
        ? ::tyloi::ScalarType::float32
        : ::tyloi::ScalarType::float16;
    const auto shape = gate.sizes().vec();
    tyloi::cuda::kernels::gelu_mul(
        ::tyloi::make_contiguous_view(
            gate.data_ptr(), shape, dtype,
            ::tyloi::Device{::tyloi::DeviceType::cuda, gate.get_device()}),
        ::tyloi::make_contiguous_view(
            up.data_ptr(), shape, dtype,
            ::tyloi::Device{::tyloi::DeviceType::cuda, up.get_device()}),
        ::tyloi::make_contiguous_view(
            out.data_ptr(), shape, dtype,
            ::tyloi::Device{::tyloi::DeviceType::cuda, out.get_device()}),
        mfq_current_cuda_stream());
    return out;
}

::tyloi::DenseTensor relu_squared_cuda(::tyloi::DenseTensor input)
{
    TYLOINFER_RUNTIME_CHECK(input.is_cuda() && input.is_contiguous(),
        "relu_squared: input must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(
        input.scalar_type() == ::tyloi::kFloat32 ||
        input.scalar_type() == ::tyloi::kFloat16 ||
        input.scalar_type() == ::tyloi::kBFloat16,
        "relu_squared: dtype must be f32, f16, or bf16");
    auto out = ::tyloi::cuda::empty_like(input);
    if (input.numel() == 0) return out;
    tyloi::cuda::activation_raw::relu_squared(input.data_ptr(), out.data_ptr(), input.numel(), raw_type(input.scalar_type()), mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<::tyloi::DenseTensor> linear_gate_beta_cuda(
    ::tyloi::DenseTensor alpha, ::tyloi::DenseTensor beta, ::tyloi::DenseTensor dt_bias, ::tyloi::DenseTensor a_log,
    bool stable_softplus)
{
    TYLOINFER_RUNTIME_CHECK(alpha.is_cuda() && beta.is_cuda(), "linear_gate_beta: alpha/beta must be cuda tensors");
    TYLOINFER_RUNTIME_CHECK(dt_bias.is_cuda() && a_log.is_cuda(), "linear_gate_beta: dt_bias/a_log must be cuda tensors");
    TYLOINFER_RUNTIME_CHECK(alpha.dim() == 3 && beta.dim() == 3, "linear_gate_beta: alpha/beta must be [B,T,V]");
    TYLOINFER_RUNTIME_CHECK(alpha.sizes() == beta.sizes(), "linear_gate_beta: alpha/beta shape mismatch");
    TYLOINFER_RUNTIME_CHECK(alpha.scalar_type() == beta.scalar_type(), "linear_gate_beta: alpha/beta dtype mismatch");
    TYLOINFER_RUNTIME_CHECK(alpha.scalar_type() == ::tyloi::kFloat32 || alpha.scalar_type() == ::tyloi::kFloat16,
                "linear_gate_beta: alpha/beta dtype must be f32 or f16");
    TYLOINFER_RUNTIME_CHECK(dt_bias.is_contiguous() && a_log.is_contiguous(), "linear_gate_beta: dt_bias/a_log must be contiguous");
    TYLOINFER_RUNTIME_CHECK(dt_bias.scalar_type() == ::tyloi::kFloat32 && a_log.scalar_type() == ::tyloi::kFloat32,
                "linear_gate_beta: dt_bias/a_log must be f32");
    int64_t B = alpha.size(0);
    int64_t T = alpha.size(1);
    int64_t V = alpha.size(2);
    TYLOINFER_RUNTIME_CHECK(dt_bias.numel() == V && a_log.numel() == V, "linear_gate_beta: parameter size mismatch");

    auto opts = ::tyloi::TensorOptions().device(alpha.device()).dtype(::tyloi::kFloat32);
    auto gate_t = ::tyloi::cuda::empty({B, V, T}, opts);
    auto beta_t = ::tyloi::cuda::empty({B, V, T}, opts);
    const auto dtype = alpha.scalar_type() == ::tyloi::kFloat32
        ? ::tyloi::ScalarType::float32
        : ::tyloi::ScalarType::float16;
    const auto device = ::tyloi::Device{::tyloi::DeviceType::cuda, alpha.get_device()};
    ::tyloi::TensorView alpha_view;
    alpha_view.data = alpha.data_ptr();
    alpha_view.rank = 3;
    alpha_view.scalar_type = dtype;
    alpha_view.device = device;
    ::tyloi::TensorView beta_view = alpha_view;
    beta_view.data = beta.data_ptr();
    for (int index = 0; index < 3; ++index) {
        alpha_view.sizes[index] = alpha.size(index);
        alpha_view.strides[index] = alpha.stride(index);
        beta_view.sizes[index] = beta.size(index);
        beta_view.strides[index] = beta.stride(index);
    }
    const std::array<std::int64_t, 1> parameter_shape = {V};
    const std::array<std::int64_t, 3> output_shape = {B, V, T};
    tyloi::cuda::kernels::linear_gate_beta(
        alpha_view,
        beta_view,
        ::tyloi::make_contiguous_view(
            dt_bias.data_ptr(), parameter_shape, ::tyloi::ScalarType::float32, device),
        ::tyloi::make_contiguous_view(
            a_log.data_ptr(), parameter_shape, ::tyloi::ScalarType::float32, device),
        ::tyloi::make_contiguous_view(
            gate_t.data_ptr(), output_shape, ::tyloi::ScalarType::float32, device),
        ::tyloi::make_contiguous_view(
            beta_t.data_ptr(), output_shape, ::tyloi::ScalarType::float32, device),
        mfq_current_cuda_stream(), stable_softplus);
    return {gate_t, beta_t};
}


namespace tyloi::cuda::kernels {
void silu_mul(const TensorView& gate, const TensorView& up, const TensorView& output, cudaStream_t stream) {
    if (!gate.is_contiguous() || !up.is_contiguous() || !output.is_contiguous() ||
        gate.numel() != up.numel() || gate.numel() != output.numel() ||
        gate.scalar_type != up.scalar_type || gate.scalar_type != output.scalar_type)
        throw std::invalid_argument("native silu_mul tensor mismatch");
    if (gate.numel() == 0) return;
    activation_raw::silu_mul(gate.data, up.data, output.data, gate.numel(), raw_type(gate.scalar_type), stream);
}
void gelu_mul(const TensorView& gate, const TensorView& up, const TensorView& output, cudaStream_t stream) {
    if (!gate.is_contiguous() || !up.is_contiguous() || !output.is_contiguous() ||
        gate.numel() != up.numel() || gate.numel() != output.numel() ||
        gate.scalar_type != up.scalar_type || gate.scalar_type != output.scalar_type)
        throw std::invalid_argument("native gelu_mul tensor mismatch");
    if (gate.numel() == 0) return;
    activation_raw::gelu_mul(gate.data, up.data, output.data, gate.numel(), raw_type(gate.scalar_type), stream);
}
void linear_gate_beta(const TensorView& alpha, const TensorView& beta, const TensorView& bias,
    const TensorView& log, const TensorView& gate, const TensorView& beta_out, cudaStream_t stream, bool stable) {
    if (alpha.rank != 3 || beta.rank != 3 || alpha.scalar_type != beta.scalar_type ||
        alpha.sizes != beta.sizes || bias.scalar_type != ScalarType::float32 ||
        log.scalar_type != ScalarType::float32 || gate.scalar_type != ScalarType::float32 ||
        beta_out.scalar_type != ScalarType::float32)
        throw std::invalid_argument("native linear_gate_beta tensor mismatch");
    const auto B = alpha.sizes[0], T = alpha.sizes[1], V = alpha.sizes[2];
    if (bias.numel() != V || log.numel() != V || gate.numel() != alpha.numel() || beta_out.numel() != alpha.numel())
        throw std::invalid_argument("native linear_gate_beta shape mismatch");
    if (alpha.numel() == 0) return;
    activation_raw::linear_gate_beta(alpha.data, beta.data, bias.data_as<float>(), log.data_as<float>(),
        gate.data_as<float>(), beta_out.data_as<float>(), B, T, V, alpha.strides.data(), beta.strides.data(),
        raw_type(alpha.scalar_type), stable, stream);
}
} // namespace tyloi::cuda::kernels
