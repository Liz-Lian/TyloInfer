#include "rms.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
mfq::Tensor mfq::cuda::ops::rms_norm(mfq::Tensor input, const mfq::Tensor& weight,
                                     double eps, double offset, mfq::ScalarType output_type) {
    mfq_tensor_backend::Tensor value(input);
    auto flat = value.reshape({-1, value.size(-1)});
    if (output_type == mfq::ScalarType::float16 && value.scalar_type() == mfq::ScalarType::float16)
        return rms_norm_f16_cuda(flat.contiguous(), mfq_tensor_backend::Tensor(weight), eps, offset)
            .reshape(value.sizes());
    auto normalized = qwen_rms_norm(flat.to(mfq_tensor_backend::kFloat32),
                                    mfq_tensor_backend::Tensor(weight), eps, offset);
    if (output_type == mfq::ScalarType::float16)
        normalized = normalized.to(mfq_tensor_backend::kFloat16).contiguous();
    return normalized.reshape(value.sizes());
}
mfq::ops::ResidualNorms mfq::cuda::ops::residual_norms(
    mfq::Tensor residual, mfq::Tensor attention, const mfq::Tensor& attention_weight,
    const mfq::Tensor& dense_weight, const mfq::Tensor& router_weight,
    const mfq::Tensor& expert_weight, double eps) {
    using T = mfq_tensor_backend::Tensor;
    auto values = gemma4_attn_residual_pre_norms_f16_cuda(
        T(residual), T(attention), T(attention_weight), T(dense_weight),
        T(router_weight), T(expert_weight), eps);
    return {values.at(0), values.at(1), values.at(2), values.at(3)};
}
mfq::Tensor mfq::cuda::ops::merge_norms(
    mfq::Tensor dense, mfq::Tensor experts, mfq::Tensor residual,
    const mfq::Tensor& dense_weight, const mfq::Tensor& expert_weight,
    const mfq::Tensor& output_weight, const mfq::Tensor& scale, double eps) {
    using T = mfq_tensor_backend::Tensor;
    return gemma4_ffn_merge_f16_cuda(T(dense), T(experts), T(residual), T(dense_weight),
                                    T(expert_weight), T(output_weight), T(scale), eps);
}
#endif

mfq_tensor_backend::Tensor qwen_rms_norm(mfq_tensor_backend::Tensor x,
                                         mfq_tensor_backend::Tensor weight, double eps,
                                         double weight_offset) {
    if (!x.is_cuda()) {
        auto xf = x.contiguous().to(mfq_tensor_backend::kFloat32);
        auto wf = weight.contiguous().to(mfq_tensor_backend::kFloat32);
        if (weight_offset != 0.0)
            wf = wf + weight_offset;
        auto inverse = mfq_tensor_backend::rsqrt(xf.square().mean(-1, true) + eps);
        return (xf * inverse * wf).contiguous();
    }
    return rms_norm_offset_cuda(x, weight, eps, weight_offset);
}

mfq_tensor_backend::Tensor qwen_rms_norm_bf16(bool fused,
                                              mfq_tensor_backend::Tensor x,
                                              mfq_tensor_backend::Tensor weight, double eps,
                                              double weight_offset) {
    auto input = x.contiguous().to(mfq_tensor_backend::kBFloat16);
    if (input.is_cuda() && fused) {
        return qwen_rms_norm_bf16_cuda(input, weight.contiguous(), eps, weight_offset);
    }
    auto xf = input.to(mfq_tensor_backend::kFloat32);
    auto inverse = mfq_tensor_backend::rsqrt(xf.square().mean(-1, true) + eps);
    auto normalized = (xf * inverse).to(mfq_tensor_backend::kBFloat16);
    auto scale = weight.contiguous().to(mfq_tensor_backend::kBFloat16);
    if (weight_offset != 0.0)
        scale = scale + weight_offset;
    return (scale * normalized).contiguous();
}
