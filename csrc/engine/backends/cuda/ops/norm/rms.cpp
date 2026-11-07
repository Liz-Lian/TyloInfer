#include "rms.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"

#include <initializer_list>
#include <stdexcept>

#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/executor.h"
tyloi::DenseTensor tyloi::cuda::ops::rms_norm(tyloi::DenseTensor input, const tyloi::DenseTensor& weight,
                                     double eps, double offset, tyloi::ScalarType output_type) {
    MfqCudaGuard guard(input.device().index);
    ::tyloi::DenseTensor value(input);
    auto flat = value.reshape({-1, value.size(-1)});
    if (output_type == tyloi::ScalarType::bfloat16)
        return qwen_rms_norm_bf16(value.is_cuda() && Executor::current().attention_options().fused_norm,
            flat, weight, eps, offset).reshape(value.sizes());
    if (output_type == tyloi::ScalarType::float16 && value.scalar_type() == tyloi::ScalarType::float16)
        return rms_norm_f16_cuda(flat.contiguous(), weight, eps, offset)
            .reshape(value.sizes());
    auto normalized = qwen_rms_norm(flat.to(::tyloi::kFloat32),
                                    weight, eps, offset);
    if (output_type == tyloi::ScalarType::float16)
        normalized = normalized.to(::tyloi::kFloat16).contiguous();
    return normalized.reshape(value.sizes());
}
std::array<tyloi::DenseTensor, 3> tyloi::cuda::ops::qkv_norm(std::array<tyloi::DenseTensor, 3> values,
    std::array<const tyloi::DenseTensor*, 3> weights, double eps, double offset, tyloi::ScalarType output_type) {
    using T = ::tyloi::DenseTensor;
    T q(values[0]), k(values[1]), v(values[2]);
    const auto* qw = weights[0]; const auto* kw = weights[1]; const auto* vw = weights[2];
    const bool bf16 = output_type == tyloi::ScalarType::bfloat16;
    const bool fused = Executor::current().attention_options().fused_norm;
    if (bf16 && fused && qw && qw->defined() && kw && kw->defined() &&
        q.scalar_type() == ::tyloi::kBFloat16 && k.scalar_type() == ::tyloi::kBFloat16) {
        if (values[0].device() != values[1].device())
            throw std::invalid_argument("paired norm tensor devices do not match");
        auto normalized = tyloi::ops::measure("full.qk_norm", values[0].device(), [&] {
            MfqCudaGuard guard(values[0].device().index);
            return qwen_rms_norm_pair_bf16_cuda(q, k, *qw, *kw, eps, offset);
        });
        q = normalized[0].reshape_as(q); k = normalized[1].reshape_as(k);
    } else if (!bf16 && qw && qw->defined() && kw && kw->defined() &&
        q.scalar_type() == ::tyloi::kFloat16 && k.scalar_type() == ::tyloi::kFloat16) {
        if (values[0].device() != values[1].device())
            throw std::invalid_argument("paired norm tensor devices do not match");
        auto normalized = tyloi::ops::measure("full.qk_norm", values[0].device(), [&] {
            MfqCudaGuard guard(values[0].device().index);
            return rms_norm_pair_f16_f32_offset_cuda(q, k, *qw, *kw, eps, offset);
        });
        q = normalized[0].reshape_as(q); k = normalized[1].reshape_as(k);
    } else {
        if (qw && qw->defined()) q = tyloi::ops::measure("full.q_norm", values[0].device(), [&] {
            MfqCudaGuard guard(values[0].device().index);
            return bf16 ? qwen_rms_norm_bf16(fused, q.reshape({-1, q.size(-1)}), *qw, eps, offset).reshape_as(q)
                        : qwen_rms_norm(q.reshape({-1, q.size(-1)}).to(::tyloi::kFloat32), *qw, eps, offset).reshape_as(q);
        });
        if (kw && kw->defined()) k = tyloi::ops::measure("full.k_norm", values[1].device(), [&] {
            MfqCudaGuard guard(values[1].device().index);
            return bf16 ? qwen_rms_norm_bf16(fused, k.reshape({-1, k.size(-1)}), *kw, eps, offset).reshape_as(k)
                        : qwen_rms_norm(k.reshape({-1, k.size(-1)}).to(::tyloi::kFloat32), *kw, eps, offset).reshape_as(k);
        });
    }
    if (vw && vw->defined()) v = tyloi::ops::measure("full.v_norm", values[2].device(), [&] {
        MfqCudaGuard guard(values[2].device().index);
        return bf16 ? qwen_rms_norm_bf16(fused, v.reshape({-1, v.size(-1)}), *vw, eps, offset).reshape_as(v)
                    : qwen_rms_norm(v.reshape({-1, v.size(-1)}).to(::tyloi::kFloat32), *vw, eps, offset).reshape_as(v);
    });
    return {q, k, v};
}
std::optional<tyloi::ops::ResidualNorms> tyloi::cuda::ops::residual_norms(
    tyloi::DenseTensor residual, tyloi::DenseTensor attention, const tyloi::DenseTensor& attention_weight,
    const tyloi::DenseTensor& dense_weight, const tyloi::DenseTensor& router_weight,
    const tyloi::DenseTensor& expert_weight, double eps) {
    if (!Executor::current().mode().gemma4_fused_norms) return std::nullopt;
    for (const tyloi::DenseTensor* value : std::initializer_list<const tyloi::DenseTensor*>{
             &attention, &attention_weight, &dense_weight, &router_weight, &expert_weight}) {
        if (!residual.defined()) throw std::invalid_argument("undefined tensor");
        if (!residual.device().is_cpu() && !residual.device().is_cuda())
            throw std::invalid_argument("unsupported tensor device");
        if (!value->defined()) throw std::invalid_argument("undefined tensor");
        if (!value->device().is_cpu() && !value->device().is_cuda())
            throw std::invalid_argument("unsupported tensor device");
        if (residual.device() != value->device()) throw std::invalid_argument("tensor devices do not match");
    }
    MfqCudaGuard guard(residual.device().index);
    using T = ::tyloi::DenseTensor;
    auto values = tyloi::ops::measure("gemma.attn_residual_pre_norms", residual.device(), [&] {
        auto flat_residual = T(residual).reshape({-1, residual.size(-1)});
        auto flat_attention = T(attention).reshape(flat_residual.sizes());
        return gemma4_attn_residual_pre_norms_f16_cuda(
            flat_residual, flat_attention, T(attention_weight), T(dense_weight),
            T(router_weight), T(expert_weight), eps);
    });
    return tyloi::ops::ResidualNorms{std::make_shared<T>(values.at(0)), std::make_shared<T>(values.at(1)),
                                     std::make_shared<T>(values.at(2)), std::make_shared<T>(values.at(3))};
}
tyloi::DenseTensor tyloi::cuda::ops::merge_norms(
    tyloi::DenseTensor dense, tyloi::DenseTensor experts, tyloi::DenseTensor residual,
    const tyloi::DenseTensor& dense_weight, const tyloi::DenseTensor& expert_weight,
    const tyloi::DenseTensor& output_weight, const tyloi::DenseTensor& scale, double eps) {
    MfqCudaGuard guard(dense.device().index);
    using T = ::tyloi::DenseTensor;
    return gemma4_ffn_merge_f16_cuda(T(dense), T(experts), T(residual), T(dense_weight),
                                    T(expert_weight), T(output_weight), T(scale), eps);
}
#endif

::tyloi::DenseTensor qwen_rms_norm(::tyloi::DenseTensor x,
#ifdef TYLOINFER_NATIVE_CUDA
                                         const ::tyloi::Tensor& weight_tensor, double eps,
#else
                                         ::tyloi::DenseTensor weight, double eps,
#endif
                                         double weight_offset) {
    if (!x.is_cuda()) {
        auto xf = x.contiguous().to(::tyloi::kFloat32);
#ifdef TYLOINFER_NATIVE_CUDA
        const auto& weight = dynamic_cast<const ::tyloi::DenseTensor&>(weight_tensor);
#endif
        auto wf = weight.contiguous().to(::tyloi::kFloat32);
        if (weight_offset != 0.0)
            wf = wf + weight_offset;
        auto inverse = ::tyloi::cuda::rsqrt(xf.square().mean(-1, true) + eps);
        return (xf * inverse * wf).contiguous();
    }
#ifdef TYLOINFER_NATIVE_CUDA
    const auto& weight = dynamic_cast<const ::tyloi::DenseTensor&>(weight_tensor);
#endif
    return rms_norm_offset_cuda(x, weight, eps, weight_offset);
}

::tyloi::DenseTensor qwen_rms_norm_bf16(bool fused,
                                              ::tyloi::DenseTensor x,
                                              ::tyloi::DenseTensor weight, double eps,
                                              double weight_offset) {
    auto input = x.contiguous().to(::tyloi::kBFloat16);
    if (input.is_cuda() && fused) {
        return qwen_rms_norm_bf16_cuda(input, weight.contiguous(), eps, weight_offset);
    }
    auto xf = input.to(::tyloi::kFloat32);
    auto inverse = ::tyloi::cuda::rsqrt(xf.square().mean(-1, true) + eps);
    auto normalized = (xf * inverse).to(::tyloi::kBFloat16);
    auto scale = weight.contiguous().to(::tyloi::kBFloat16);
    if (weight_offset != 0.0)
        scale = scale + weight_offset;
    return (scale * normalized).contiguous();
}
