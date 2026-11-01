#pragma once

#include <nlohmann/json_fwd.hpp>

#include <string>

namespace mfq::models {

std::string gguf_name(const std::string& name);
nlohmann::json gguf_config(const nlohmann::json& meta,
                           const nlohmann::json& shapes,
                           const std::string& arch);

} // namespace mfq::models
