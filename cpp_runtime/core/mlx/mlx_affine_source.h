#pragma once

#include "tyloinfer/hf_safetensors_source.h"
#include "tyloinfer/mlx_affine.h"
#include "nlohmann/json.hpp"

namespace mfq {

struct MlxAffineBinding {
    MlxAffineLayout layout;
    std::vector<std::byte> prefix;
    std::string scales, biases;
};

MlxAffineBinding mlx_affine_binding(const HfSafetensorsSource& source,
    const nlohmann::json& quantization, const TensorMetadata& values);

} // namespace mfq
