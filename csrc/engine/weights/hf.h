#pragma once

#include "engine/weights/source/hf.h"

#include <nlohmann/json_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mfq {

class Safetensors;

namespace hf {

Record prepare(
    const Safetensors& files,
    const TensorMetadata& values,
    std::string name,
    const std::vector<std::int64_t>& dense_shape,
    const std::vector<std::string>& scale_names,
    const nlohmann::json& quantization);

Record experts(std::string name, const std::map<std::size_t, const Record*>& records);

} // namespace hf
} // namespace mfq
