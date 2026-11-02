#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mfq::source {

class Assets final {
public:
    void add(std::string name, const std::filesystem::path& path);
    void add(std::string name, std::string payload);
    void add_directory(const std::filesystem::path& directory);

    bool has(std::string_view name) const noexcept;
    std::vector<std::byte> read(std::string_view name) const;
    std::vector<std::string> names() const;

private:
    std::unordered_map<std::string, std::filesystem::path> paths_;
    std::unordered_map<std::string, std::vector<std::byte>> inline_;
};

} // namespace mfq::source
