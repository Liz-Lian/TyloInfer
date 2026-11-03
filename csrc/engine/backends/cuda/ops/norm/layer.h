#pragma once

#include "backend.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/ops.h"
#endif

#include <stdexcept>
#include <utility>

namespace mfq::cuda {

inline mfq_tensor_backend::Tensor vision_layer_norm(
        mfq_tensor_backend::Tensor input, const mfq_tensor_backend::Tensor &weight,
        const mfq_tensor_backend::Tensor &bias, double epsilon) {
    const auto dtype = input.scalar_type();
    auto output = mfq_tensor_backend::layer_norm(input, {input.size(-1)}, weight.to(dtype),
                                                 bias.to(dtype), epsilon);
    return output.to(dtype).contiguous();
}

} // namespace mfq::cuda

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
namespace mfq::cuda::ops {
inline mfq::Tensor layer_norm(mfq::Tensor input, const mfq::Tensor& weight,
                              const mfq::Tensor& bias, double epsilon) {
    MfqCudaGuard guard(input.device().index);
    return mfq_tensor_backend::layer_norm(mfq_tensor_backend::Tensor(input), {input.size(-1)},
        mfq_tensor_backend::Tensor(weight), mfq_tensor_backend::Tensor(bias), epsilon);
}
}
#endif

namespace mfq::cuda::vision {

using Tensor = mfq_tensor_backend::Tensor;

struct RmsNorm {
    Tensor weight;
    double epsilon = 1e-6;

    template <class Visitor>
    void visit_tensors(Visitor visit) {
        visit(weight);
    }

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    mfq::Tensor operator()(const mfq::Tensor& input) const {
#else
    Tensor operator()(const Tensor& input) const {
#endif
        if (input.dim() == 0 || input.size(-1) != weight.size(0)) {
            throw std::runtime_error("vision RMSNorm input width mismatch");
        }
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        return mfq::ops::contiguous(mfq::ops::cast(
            mfq::ops::rms_norm(input, weight, epsilon, 0.0), input.scalar_type()));
#else
        const auto shape = input.sizes().vec();
        auto normalized = rms_norm_cuda(
            input.reshape({-1, shape.back()}).to(mfq_tensor_backend::kFloat32),
            weight,
            epsilon);
        return normalized.reshape(shape)
            .to(input.scalar_type())
            .contiguous();
#endif
    }
};

struct LayerNorm {
    Tensor weight, bias;
    double epsilon = 1e-6;

    template <class Visitor>
    void visit_tensors(Visitor visit) {
        visit(weight);
        visit(bias);
    }

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    mfq::Tensor operator()(const mfq::Tensor& input) const {
#else
    Tensor operator()(const Tensor& input) const {
#endif
        if (input.dim() == 0 || input.size(-1) != weight.size(0))
            throw std::runtime_error("grid-ViT LayerNorm input width mismatch");
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        return mfq::ops::layer_norm(input, weight, bias, epsilon);
#else
        return mfq_tensor_backend::layer_norm(input, {weight.size(0)}, weight, bias, epsilon);
#endif
    }
};

} // namespace mfq::cuda::vision
