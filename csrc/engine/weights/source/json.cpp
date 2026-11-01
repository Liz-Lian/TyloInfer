#include "engine/weights/source/json.h"

#include "nlohmann/json.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace mfq::source {
namespace {

constexpr std::uint64_t kMaxJsonSidecarBytes = std::uint64_t{256} << 20;

} // namespace

std::vector<std::byte> read_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    stream.seekg(0, std::ios::end);
    const auto end = stream.tellg();
    if (end < 0 || static_cast<std::uintmax_t>(end) >
                       std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("cannot size " + path.string());
    }
    std::vector<std::byte> result(static_cast<std::size_t>(end));
    stream.seekg(0, std::ios::beg);
    stream.read(reinterpret_cast<char*>(result.data()),
                static_cast<std::streamsize>(result.size()));
    if (!stream && !result.empty()) {
        throw std::runtime_error("cannot read " + path.string());
    }
    return result;
}

std::string read_text(const std::filesystem::path& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size > kMaxJsonSidecarBytes) {
        throw std::runtime_error("JSON sidecar is too large: " + path.string());
    }
    const auto bytes = read_file(path);
    std::string result(bytes.size(), '\0');
    if (!bytes.empty()) {
        std::memcpy(result.data(), bytes.data(), bytes.size());
    }
    return result;
}

nlohmann::json parse_json(
    std::string_view payload,
    const std::filesystem::path& path) {
    try {
        return nlohmann::json::parse(payload.begin(), payload.end());
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("invalid JSON in " + path.string() + ": " +
                                 error.what());
    }
}

} // namespace mfq::source
