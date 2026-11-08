#pragma once

#include <nlohmann/json_fwd.hpp>

#include <string>

namespace tyloi::models {

std::string gguf_name(const std::string& name);
nlohmann::json gguf_config(const nlohmann::json& meta,
                           const nlohmann::json& shapes,
                           const std::string& arch);

} // namespace tyloi::models
