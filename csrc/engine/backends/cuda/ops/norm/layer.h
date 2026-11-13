#pragma once

#include "backend.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#include "engine/ops.h"
#include "engine/weights/fields.h"

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

namespace tyloi::cuda::ops {
inline tyloi::DenseTensor layer_norm(tyloi::DenseTensor input, const tyloi::DenseTensor& weight,
                              const tyloi::DenseTensor& bias, double epsilon) {
    MfqCudaGuard guard(input.device().index);
    return ::tyloi::cuda::layer_norm(input, {input.size(-1)},
        weight, bias, epsilon);
}
}

namespace tyloi::cuda::vision {

using DenseTensor = ::tyloi::DenseTensor;

struct RmsNorm {
    ::tyloi::Tensor weight;
    double epsilon = 1e-6;

    template <class Visitor>
    void visit_tensors(Visitor visit) const {
        if (weight && weight.defined()) ::tyloi::weights::fields::tensors(weight, visit);
    }
    void to_device(::tyloi::Device device) {
        if (weight && weight.defined()) ::tyloi::weights::fields::to_device(weight, device);
    }

    ::tyloi::Tensor operator()(const ::tyloi::Tensor& input) const {
        if (input.dim() == 0 || !weight || !weight.defined() || input.size(-1) != weight.size(0)) {
            throw std::runtime_error("vision RMSNorm input width mismatch");
        }
        return tyloi::ops::contiguous(tyloi::ops::cast(
            tyloi::ops::rms_norm(input, weight, epsilon, 0.0), input.scalar_type()));
    }
};

struct LayerNorm {
    ::tyloi::Tensor weight, bias;
    double epsilon = 1e-6;

    template <class Visitor>
    void visit_tensors(Visitor visit) const {
        if (weight && weight.defined()) ::tyloi::weights::fields::tensors(weight, visit);
        if (bias && bias.defined()) ::tyloi::weights::fields::tensors(bias, visit);
    }
    void to_device(::tyloi::Device device) {
        if (weight && weight.defined()) ::tyloi::weights::fields::to_device(weight, device);
        if (bias && bias.defined()) ::tyloi::weights::fields::to_device(bias, device);
    }

    ::tyloi::Tensor operator()(const ::tyloi::Tensor& input) const {
        if (input.dim() == 0 || !weight || !weight.defined() || input.size(-1) != weight.size(0))
            throw std::runtime_error("grid-ViT LayerNorm input width mismatch");
        if (!bias || !bias.defined())
            throw std::runtime_error("grid-ViT LayerNorm bias is undefined");
        return tyloi::ops::layer_norm(input, weight, bias, epsilon);
    }
};

} // namespace tyloi::cuda::vision
