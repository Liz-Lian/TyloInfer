#pragma once

#include <nlohmann/json_fwd.hpp>

#include <cstdint>
#include <filesystem>
#include <vector>

struct gguf_context;

namespace mfq {

bool is_gguf_file(const std::filesystem::path& path);

} // namespace mfq

namespace mfq::gguf {

nlohmann::json scalar(const gguf_context* ctx, std::int64_t id);
nlohmann::json value(const gguf_context* ctx, std::int64_t id);
std::vector<std::filesystem::path> paths(const std::filesystem::path& path, int splits);

} // namespace mfq::gguf
