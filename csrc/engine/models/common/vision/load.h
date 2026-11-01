#pragma once

#include "engine/weights/loader.h"
#include "ops/vision/vision.h"
#include "ops/vision/block.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace mfq::cuda::vision {

inline BiasLinear load_biased(const mfq::cuda::weight_loader::Loader& weights,
                              const std::string &prefix, bool with_bias = true) {
    BiasLinear result;
    result.weight = weights.projection(prefix + ".weight");
    const std::string bias_name = prefix + ".bias";
    if (with_bias && has_tensor(weights.source, bias_name)) {
        result.bias = weights.tensor(bias_name);
    }
    return result;
}

inline Affine load_affine(
    const mfq::cuda::weight_loader::Loader& weights,
    const std::string& weight_name,
    const std::string& bias_name) {
    Affine result{weights.projection(weight_name), {}};
    if (has_tensor(weights.source, bias_name)) {
        result.bias = weights.tensor(bias_name);
        if (result.bias.dim() != 1 ||
            result.bias.size(0) != result.linear.out()) {
            throw std::runtime_error(
                "vision affine bias shape mismatch: " + bias_name);
        }
    }
    return result;
}

inline RmsNorm load_rms_norm(
    const mfq::cuda::weight_loader::Loader& weights,
    const std::string& name,
    std::int64_t width,
    double epsilon) {
    auto weight = weights.tensor(name)
                      .to(mfq_tensor_backend::kFloat32)
                      .contiguous();
    if (weight.dim() != 1 || weight.size(0) != width) {
        throw std::runtime_error("vision RMSNorm width mismatch: " + name);
    }
    return {std::move(weight), epsilon};
}

} // namespace mfq::cuda::vision
