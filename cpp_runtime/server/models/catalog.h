#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace mfq::server {

class ModelCatalog {
public:
    struct Artifact {
        std::filesystem::path path;
        std::string id, name, architecture = "unknown", format, modified_at, error;
        std::uint64_t total_bytes = 0;
        std::size_t shard_count = 0, missing_shards = 0, tensor_count = 0, record_count = 0;
        std::vector<std::string> dtypes;
        std::optional<std::int64_t> context_capacity;
        bool complete = false;
    };
    struct DirectorySource { std::string id; std::filesystem::path path; };
    struct DirectoryEntry { std::string id, name; std::size_t model_file_count; };
    struct DirectoryFile { std::string name; std::uint64_t bytes; };
    struct DirectoryListing {
        std::optional<DirectoryEntry> current;
        std::filesystem::path path;
        std::optional<std::string> parent_id;
        std::vector<DirectoryFile> files;
        std::vector<DirectoryEntry> children;
    };
    explicit ModelCatalog(std::vector<std::filesystem::path> roots);
    std::vector<Artifact> list(bool refresh);
    DirectoryListing browse(const DirectorySource& source);
    std::vector<Artifact> register_directory(const DirectorySource& source);
    DirectoryListing model_directory(const std::string& model_id);
    Artifact resolve(const std::string& model);

private:
    std::set<std::filesystem::path> registered_paths() const;
    std::vector<Artifact> scan(const std::set<std::filesystem::path>& registered) const;
    void refresh();
    std::string directory_id(const std::filesystem::path& path);
    std::filesystem::path directory(const DirectorySource& source) const;
    DirectoryEntry directory_entry(const std::filesystem::path& path);
    DirectoryListing browse(const std::filesystem::path& path);

    std::mutex mutex_;
    std::vector<std::filesystem::path> roots_;
    std::string secret_;
    std::map<std::string, std::filesystem::path> directories_;
    std::vector<Artifact> artifacts_;
    bool scanned_ = false;
    std::chrono::steady_clock::time_point scanned_at_;
};

} // namespace mfq::server
