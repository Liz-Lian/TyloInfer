#pragma once

#include <filesystem>
#include <mutex>

struct sqlite3;

namespace mfq::server {

class Storage {
public:
    explicit Storage(const std::filesystem::path& path);
    ~Storage();
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    struct Access {
        std::unique_lock<std::mutex> lock;
        sqlite3* database;
    };
    // Open on first access so startup policy validation cannot create or migrate data.
    Access access();

private:
    sqlite3* database_ = nullptr;
    std::filesystem::path path_;
    std::mutex mutex_;
};

} // namespace mfq::server
