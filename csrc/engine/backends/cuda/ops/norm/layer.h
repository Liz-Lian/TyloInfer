#pragma once

#include "backend.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"

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

namespace mfq::cuda::vision {

using Tensor = mfq_tensor_backend::Tensor;

struct RmsNorm {
    Tensor weight;
    double epsilon = 1e-6;

    template <class Visitor>
    void visit_tensors(Visitor visit) {
        visit(weight);
    }

    Tensor operator()(const Tensor& input) const {
        if (input.dim() == 0 || input.size(-1) != weight.size(0)) {
            throw std::runtime_error("vision RMSNorm input width mismatch");
        }
        const auto shape = input.sizes().vec();
        auto normalized = rms_norm_cuda(
            input.reshape({-1, shape.back()}).to(mfq_tensor_backend::kFloat32),
            weight,
            epsilon);
        return normalized.reshape(shape)
            .to(input.scalar_type())
            .contiguous();
    }
};

struct LayerNorm {
    Tensor weight, bias;
    double epsilon = 1e-6;

    Tensor operator()(const Tensor& input) const {
        if (input.dim() == 0 || input.size(-1) != weight.size(0))
            throw std::runtime_error("grid-ViT LayerNorm input width mismatch");
        return mfq_tensor_backend::layer_norm(input, {weight.size(0)}, weight, bias, epsilon);
    }
};

} // namespace mfq::cuda::vision
