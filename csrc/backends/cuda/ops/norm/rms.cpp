#include "ops/norm/rms.h"
#include "mfq_cuda_norm_ops.h"

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
