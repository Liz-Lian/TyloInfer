#pragma once

#include "backends/cuda/engine/weights/load.h"
#include "ops/vision/vision.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace mfq::cuda::vision {

inline Affine load_affine(
    CudaExecutionContext& execution,
    const mfq::ModelSource& model,
    const std::string& weight_name,
    const std::string& bias_name) {
    Affine result{load_quant_linear(execution, model, weight_name), {}};
    if (has_tensor(model, bias_name)) {
        result.bias = load_dense_gpu(execution, model, bias_name);
        if (result.bias.dim() != 1 ||
            result.bias.size(0) != result.linear.out()) {
            throw std::runtime_error(
                "vision affine bias shape mismatch: " + bias_name);
        }
    }
    return result;
}

inline RmsNorm load_rms_norm(
    CudaExecutionContext& execution,
    const mfq::ModelSource& model,
    const std::string& name,
    std::int64_t width,
    double epsilon) {
    auto weight = load_dense_gpu(execution, model, name)
                      .to(mfq_tensor_backend::kFloat32)
                      .contiguous();
    if (weight.dim() != 1 || weight.size(0) != width) {
        throw std::runtime_error("vision RMSNorm width mismatch: " + name);
    }
    return {std::move(weight), epsilon};
}

} // namespace mfq::cuda::vision
