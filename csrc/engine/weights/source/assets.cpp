#include "engine/weights/source/assets.h"

#include "engine/weights/source/json.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace tyloi::source {
namespace {

constexpr std::string_view kAssetPrefix = "__mfq_asset__/";

} // namespace

void Assets::add(std::string name, const std::filesystem::path& path) {
    std::error_code asset_error;
    const auto resolved = std::filesystem::canonical(path, asset_error);
    if (asset_error || !std::filesystem::is_regular_file(resolved, asset_error) ||
        asset_error || std::filesystem::file_size(resolved, asset_error) == 0 ||
        asset_error) {
        throw std::runtime_error("invalid native HF runtime asset: " +
                                 path.string());
    }
    if (inline_.find(name) != inline_.end() ||
        !paths_.emplace(name, resolved).second) {
        throw std::runtime_error("duplicate native HF runtime asset: " +
                                 path.string());
    }
}

void Assets::add(std::string name, std::string payload) {
    std::vector<std::byte> bytes(payload.size());
    if (!payload.empty()) {
        std::memcpy(bytes.data(), payload.data(), payload.size());
    }
    if (paths_.find(name) != paths_.end() ||
        !inline_.emplace(name, std::move(bytes)).second) {
        throw std::runtime_error("duplicate native HF inline asset: " + name);
    }
}

void Assets::add_directory(const std::filesystem::path& directory) {
    std::error_code directory_error;
    if (!std::filesystem::is_directory(directory, directory_error) ||
        directory_error) return;
    const auto resolved = std::filesystem::canonical(directory, directory_error);
    if (directory_error) {
        throw std::runtime_error(
            "cannot resolve native HF runtime asset directory: " +
            directory.string());
    }
    for (std::filesystem::recursive_directory_iterator iterator(
             resolved, directory_error), end;
         !directory_error && iterator != end;
         iterator.increment(directory_error)) {
        if (!iterator->is_regular_file(directory_error) || directory_error) continue;
        const auto relative = std::filesystem::relative(
            iterator->path(), resolved, directory_error);
        if (directory_error || relative.empty() || relative.is_absolute() ||
            *relative.begin() == "..") {
            throw std::runtime_error(
                "invalid native HF runtime asset path: " +
                iterator->path().string());
        }
        add(std::string(kAssetPrefix) + relative.generic_string(),
            iterator->path());
    }
    if (directory_error) {
        throw std::runtime_error(
            "cannot enumerate native HF runtime assets: " + resolved.string());
    }
}

bool Assets::has(std::string_view name) const noexcept {
    const auto key = std::string(name);
    return inline_.find(key) != inline_.end() || paths_.find(key) != paths_.end();
}

std::vector<std::byte> Assets::read(std::string_view name) const {
    const auto key = std::string(name);
    if (const auto found = inline_.find(key); found != inline_.end()) {
        return found->second;
    }
    const auto found = paths_.find(key);
    if (found == paths_.end()) {
        throw std::runtime_error("model asset not found: " + std::string(name));
    }
    return read_file(found->second);
}

std::vector<std::string> Assets::names() const {
    std::vector<std::string> result;
    result.reserve(paths_.size() + inline_.size());
    for (const auto& [name, unused] : paths_) {
        static_cast<void>(unused);
        result.push_back(name);
    }
    for (const auto& [name, unused] : inline_) {
        static_cast<void>(unused);
        result.push_back(name);
    }
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace tyloi::source
