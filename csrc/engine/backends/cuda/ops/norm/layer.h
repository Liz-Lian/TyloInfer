#pragma once

#include "backend.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/ops.h"
#include "engine/weights/fields.h"
#endif

#include <memory>
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

#ifdef TYLOINFER_NATIVE_CUDA
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
#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<const ::tyloi::Tensor> weight;
#else
    DenseTensor weight;
#endif
    double epsilon = 1e-6;

    template <class Visitor>
    void visit_tensors(Visitor visit) const {
#ifdef TYLOINFER_NATIVE_CUDA
        if (weight && weight->defined()) fields::tensors(weight, visit);
#else
        visit(weight);
#endif
    }
    void to_device(::tyloi::Device device) {
#ifdef TYLOINFER_NATIVE_CUDA
        if (weight && weight->defined()) fields::to_device(weight, device);
#else
        if (weight.defined()) weight = weight.to(device);
#endif
    }

#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<::tyloi::Tensor> operator()(const ::tyloi::Tensor& input) const {
        if (input.dim() == 0 || !weight || !weight->defined() || input.size(-1) != weight->size(0)) {
            throw std::runtime_error("vision RMSNorm input width mismatch");
        }
        return tyloi::ops::contiguous(*tyloi::ops::cast(
            *tyloi::ops::rms_norm(input, *weight, epsilon, 0.0), input.scalar_type()));
#else
    DenseTensor operator()(const DenseTensor& input) const {
        if (input.dim() == 0 || input.size(-1) != weight.size(0)) {
            throw std::runtime_error("vision RMSNorm input width mismatch");
        }
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
#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<const ::tyloi::Tensor> weight, bias;
#else
    DenseTensor weight, bias;
#endif
    double epsilon = 1e-6;

    template <class Visitor>
    void visit_tensors(Visitor visit) const {
#ifdef TYLOINFER_NATIVE_CUDA
        if (weight && weight->defined()) fields::tensors(weight, visit);
        if (bias && bias->defined()) fields::tensors(bias, visit);
#else
        visit(weight);
        visit(bias);
#endif
    }
    void to_device(::tyloi::Device device) {
#ifdef TYLOINFER_NATIVE_CUDA
        if (weight && weight->defined()) fields::to_device(weight, device);
        if (bias && bias->defined()) fields::to_device(bias, device);
#else
        if (weight.defined()) weight = weight.to(device);
        if (bias.defined()) bias = bias.to(device);
#endif
    }

#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<::tyloi::Tensor> operator()(const ::tyloi::Tensor& input) const {
        if (input.dim() == 0 || !weight || !weight->defined() || input.size(-1) != weight->size(0))
            throw std::runtime_error("grid-ViT LayerNorm input width mismatch");
        if (!bias || !bias->defined())
            throw std::runtime_error("grid-ViT LayerNorm bias is undefined");
        return tyloi::ops::layer_norm(input, *weight, *bias, epsilon);
#else
    DenseTensor operator()(const DenseTensor& input) const {
        if (input.dim() == 0 || input.size(-1) != weight.size(0))
            throw std::runtime_error("grid-ViT LayerNorm input width mismatch");
        return ::tyloi::cuda::layer_norm(input, {weight.size(0)}, weight, bias, epsilon);
#endif
    }
};

} // namespace tyloi::cuda::vision
