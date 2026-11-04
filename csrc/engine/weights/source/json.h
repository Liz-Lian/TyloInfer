#pragma once

#include <nlohmann/json_fwd.hpp>

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace tyloi::source {

std::vector<std::byte> read_file(const std::filesystem::path& path);
std::string read_text(const std::filesystem::path& path);
nlohmann::json parse_json(
    std::string_view payload,
    const std::filesystem::path& path);

} // namespace tyloi::source
