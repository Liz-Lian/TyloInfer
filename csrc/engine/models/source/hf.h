#pragma once

#include "engine/models/source/legacy_names.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tyloi::models {

Names hf_names(
    std::string_view architecture,
    const std::vector<std::string>& stored_names,
    std::optional<std::string_view> source_map,
    std::optional<std::string_view> config);

} // namespace tyloi::models
