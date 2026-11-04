#pragma once

#include "backend.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/ops.h"
#endif

#include <stdexcept>
#include <utility>

namespace tyloi::cuda {

inline ::tyloi::DenseTensor vision_layer_norm(
        ::tyloi::DenseTensor input, const ::tyloi::DenseTensor &weight,
        const ::tyloi::DenseTensor &bias, double epsilon) {
    const auto dtype = input.scalar_type();
    auto output = ::tyloi::cuda::layer_norm(input, {input.size(-1)}, weight.to(dtype),
                                                 bias.to(dtype), epsilon);
    return output.to(dtype).contiguous();
}

} // namespace tyloi::cuda

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
namespace tyloi::cuda::ops {
inline tyloi::DenseTensor layer_norm(tyloi::DenseTensor input, const tyloi::DenseTensor& weight,
                              const tyloi::DenseTensor& bias, double epsilon) {
    MfqCudaGuard guard(input.device().index);
    return ::tyloi::cuda::layer_norm(input, {input.size(-1)},
        weight, bias, epsilon);
}
}
#endif

namespace tyloi::cuda::vision {

using DenseTensor = ::tyloi::DenseTensor;

struct RmsNorm {
    DenseTensor weight;
    double epsilon = 1e-6;

    template <class Visitor>
    void visit_tensors(Visitor visit) const {
        visit(weight);
    }
    void to_device(::tyloi::Device device) {
        if (weight.defined()) weight = weight.to(device);
    }

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    tyloi::DenseTensor operator()(const tyloi::DenseTensor& input) const {
#else
    DenseTensor operator()(const DenseTensor& input) const {
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
    DenseTensor weight, bias;
    double epsilon = 1e-6;

    template <class Visitor>
    void visit_tensors(Visitor visit) const {
        visit(weight);
        visit(bias);
    }
    void to_device(::tyloi::Device device) {
        if (weight.defined()) weight = weight.to(device);
        if (bias.defined()) bias = bias.to(device);
    }

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    tyloi::DenseTensor operator()(const tyloi::DenseTensor& input) const {
#else
    DenseTensor operator()(const DenseTensor& input) const {
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
