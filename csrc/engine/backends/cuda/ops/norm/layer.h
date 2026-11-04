#pragma once

#include "backend.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/ops.h"
#endif

#include <stdexcept>
#include <utility>

namespace tyloi::cuda {

inline ::tyloi::Tensor vision_layer_norm(
        ::tyloi::Tensor input, const ::tyloi::Tensor &weight,
        const ::tyloi::Tensor &bias, double epsilon) {
    const auto dtype = input.scalar_type();
    auto output = ::tyloi::cuda::layer_norm(input, {input.size(-1)}, weight.to(dtype),
                                                 bias.to(dtype), epsilon);
    return output.to(dtype).contiguous();
}

} // namespace tyloi::cuda

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
namespace tyloi::cuda::ops {
inline tyloi::Tensor layer_norm(tyloi::Tensor input, const tyloi::Tensor& weight,
                              const tyloi::Tensor& bias, double epsilon) {
    MfqCudaGuard guard(input.device().index);
    return ::tyloi::cuda::layer_norm(input, {input.size(-1)},
        weight, bias, epsilon);
}
}
#endif

namespace tyloi::cuda::vision {

using Tensor = ::tyloi::Tensor;

struct RmsNorm {
    Tensor weight;
    double epsilon = 1e-6;

    template <class Visitor>
    void visit_tensors(Visitor visit) {
        visit(weight);
    }

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    tyloi::Tensor operator()(const tyloi::Tensor& input) const {
#else
    Tensor operator()(const Tensor& input) const {
#endif
        if (input.dim() == 0 || input.size(-1) != weight.size(0)) {
            throw std::runtime_error("vision RMSNorm input width mismatch");
        }
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        return tyloi::ops::contiguous(tyloi::ops::cast(
            tyloi::ops::rms_norm(input, weight, epsilon, 0.0), input.scalar_type()));
#else
        const auto shape = input.sizes().vec();
        auto normalized = rms_norm_cuda(
            input.reshape({-1, shape.back()}).to(::tyloi::kFloat32),
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
    tyloi::Tensor operator()(const tyloi::Tensor& input) const {
#else
    Tensor operator()(const Tensor& input) const {
#endif
        if (input.dim() == 0 || input.size(-1) != weight.size(0))
            throw std::runtime_error("grid-ViT LayerNorm input width mismatch");
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        return tyloi::ops::layer_norm(input, weight, bias, epsilon);
#else
        return ::tyloi::cuda::layer_norm(input, {weight.size(0)}, weight, bias, epsilon);
#endif
    }
};

} // namespace tyloi::cuda::vision
