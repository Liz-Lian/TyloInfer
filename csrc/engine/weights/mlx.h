#pragma once

#include "engine/weights/encoding/mlx_affine.h"
#include "engine/weights/source/safetensors.h"
#include "nlohmann/json.hpp"

namespace mfq {

struct MlxAffineBinding {
    MlxAffineLayout layout;
    std::vector<std::byte> prefix;
    std::string scales, biases;
};

MlxAffineBinding mlx_affine_binding(const Safetensors& source,
    const nlohmann::json& quantization, const TensorMetadata& values);

} // namespace mfq
