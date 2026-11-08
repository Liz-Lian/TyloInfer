#include "mfq.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace tyloi::models {
namespace {

bool is_table(const archive::Record& record) {
    std::string_view name = record.tensor.name;
    constexpr std::string_view marker = ".tq_row_shard.";
    const auto shard = name.rfind(marker);
    if (shard != std::string_view::npos && shard + marker.size() < name.size() &&
        std::all_of(name.begin() + shard + marker.size(), name.end(),
                    [](char c) { return c >= '0' && c <= '9'; })) {
        name = name.substr(0, shard);
    }
    for (const std::string_view suffix : {
             ".associative_memory.embedding.weight",
             ".engram.embed.weight"}) {
        if (name.size() >= suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            return true;
        }
    }
    return false;
}

} // namespace

void engram_shards(archive::Shards& parts) {
    if (parts.complete) return;
    static const std::regex pattern(archive::shard_pattern, std::regex::icase);
    static const std::regex engram_pattern(R"(^(.*)-Engram-.+$)", std::regex::icase);
    std::unordered_map<std::uint64_t, std::filesystem::path> shared;
    std::vector<std::filesystem::path> directories{
        parts.entry.parent_path(), parts.entry.parent_path() / "Engram",
        parts.entry.parent_path().parent_path() / "Engram"};
    std::sort(directories.begin(), directories.end());
    directories.erase(std::unique(directories.begin(), directories.end()), directories.end());
    const auto same_suffix = [&](const std::ssub_match& candidate) {
        auto value = candidate.str();
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return value == parts.suffix;
    };
    for (const auto& directory : directories) {
        if (!std::filesystem::is_directory(directory)) continue;
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            std::smatch candidate, family;
            const auto name = entry.path().filename().string();
            if (!entry.is_regular_file() ||
                !std::regex_match(name, candidate, pattern)) continue;
            const auto shared_stem = candidate[1].str();
            if (!std::regex_match(shared_stem, family, engram_pattern) ||
                parts.stem.compare(0, family[1].length() + 1,
                                   family[1].str() + "-") != 0 ||
                std::stoull(candidate[3].str()) != parts.paths.size() ||
                !same_suffix(candidate[4])) continue;
            const auto index = std::stoull(candidate[2].str());
            if (index < 1 || index > parts.paths.size() || parts.found[index - 1]) continue;
            if (!shared.emplace(index, entry.path()).second)
                throw std::runtime_error("ambiguous shared Engram shard " + std::to_string(index));
        }
    }
    for (const auto& [index, candidate] : shared) parts.paths[index - 1] = candidate;
}

void check_engram(
    const std::filesystem::path& path,
    const std::vector<archive::Record>& records) {
    static const std::regex shared_name(
        R"(^.*-Engram-.+-[0-9]{5}-of-[0-9]{5}\.(mfq|tq)$)",
        std::regex::icase);
    if (!std::regex_match(path.filename().string(), shared_name)) return;
    bool tables = !records.empty();
    for (const auto& record : records) {
        if (!is_table(record)) {
            tables = false;
            break;
        }
    }
    if (!tables) {
        throw std::runtime_error(
            "shared Engram shard contains non-table records: " + path.string());
    }
}

} // namespace tyloi::models
