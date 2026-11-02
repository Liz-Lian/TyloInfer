#include "storage.h"
#include "instance.h"
#include "sqlite.h"
#include "schema.h"

#include <sqlite3.h>
#include <stdexcept>
#include <utility>

namespace mfq::server {
namespace {

using json = nlohmann::json;
constexpr int schema_version = 19;
constexpr const char* key_columns = "id,name,prefix,scopes_json,role,expires_at,revoked_at,created_at,last_used_at";

using sqlite::Statement;
using sqlite::check;
using sqlite::execute;

bool table_exists(sqlite3* database, const char* name) {
    Statement query(database, "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?");
    query.bind(1, name);
    return query.step();
}

bool column_exists(sqlite3* database, const char* table, const char* column) {
    const auto sql = "PRAGMA table_info(" + std::string(table) + ")";
    Statement query(database, sql.c_str());
    while (query.step()) if (query.column(1) == column) return true;
    return false;
}

void validate_key_schema(sqlite3* database) {
    const auto sql = "SELECT " + std::string(key_columns) + ",key_hash FROM api_keys LIMIT 0";
    Statement keys(database, sql.c_str());
    keys.step();
}

void initialize(sqlite3* database) {
    int previous = 0;
    if (table_exists(database, "schema_meta")) {
        Statement query(database, "SELECT value FROM schema_meta WHERE key='schema_version'");
        if (!query.step()) throw std::runtime_error("database has no schema version");
        const auto value = query.column(0).get<std::string>();
        for (int version = 1; version <= schema_version; ++version)
            if (value == std::to_string(version)) previous = version;
        if (!previous) throw std::runtime_error("unsupported database schema " + value);
    } else {
        Statement query(database, "SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' LIMIT 1");
        if (query.step()) throw std::runtime_error("unrecognized database without schema_meta");
    }
    if (previous == schema_version) validate_key_schema(database);
    {
        Statement journal(database, "PRAGMA journal_mode=WAL");
        if (!journal.step() || journal.column(0) != "wal") throw std::runtime_error("SQLite refused WAL mode");
    }
    if (previous == schema_version) return;
    execute(database, "BEGIN IMMEDIATE");
    try {
        execute(database, schema_sql);
        if (previous && previous < schema_version) {
            for (const auto* column : {"performance_json", "output_message_id", "settings_json"}) {
                if (!column_exists(database, "responses", column))
                    execute(database, ("ALTER TABLE responses ADD COLUMN " + std::string(column) + " TEXT").c_str());
            }
            if (!column_exists(database, "api_keys", "role"))
                execute(database, "ALTER TABLE api_keys ADD COLUMN role TEXT");
            if (!column_exists(database, "generation_presets", "metadata_json"))
                execute(database, "ALTER TABLE generation_presets ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'");
            if (!column_exists(database, "jobs", "archived_at"))
                execute(database, "ALTER TABLE jobs ADD COLUMN archived_at TEXT");
            if (!column_exists(database, "jobs", "progress_data_json"))
                execute(database, "ALTER TABLE jobs ADD COLUMN progress_data_json TEXT NOT NULL DEFAULT '{}'");
            execute(database, R"SQL(
                CREATE TABLE generation_presets_nullable (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, model TEXT, mode TEXT,
                    settings_json TEXT NOT NULL, context_size INTEGER CHECK(context_size >= 512),
                    metadata_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                INSERT INTO generation_presets_nullable
                    SELECT id,name,model,mode,settings_json,context_size,metadata_json,created_at,updated_at
                    FROM generation_presets;
                DROP TABLE generation_presets;
                ALTER TABLE generation_presets_nullable RENAME TO generation_presets;
            )SQL");
        }
        validate_key_schema(database);
        Statement version(database, "INSERT OR REPLACE INTO schema_meta(key,value) VALUES('schema_version',?)");
        version.bind(1, std::to_string(schema_version));
        version.step();
        execute(database, "COMMIT");
    } catch (...) {
        sqlite3_exec(database, "ROLLBACK", nullptr, nullptr, nullptr);
        throw;
    }
}

} // namespace

Storage::Storage(const std::filesystem::path& path) : path_(path) {}

Storage::Access Storage::access() {
    std::unique_lock guard(mutex_);
    if (database_) return {std::move(guard), database_};
    std::filesystem::create_directories(path_.parent_path());
    const auto result = sqlite3_open_v2(path_utf8(path_).c_str(), &database_,
        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, nullptr);
    try {
        check(database_, result);
        check(database_, sqlite3_busy_timeout(database_, 5000));
        execute(database_, "PRAGMA foreign_keys=ON");
        initialize(database_);
    } catch (...) {
        sqlite3_close(database_);
        database_ = nullptr;
        throw;
    }
    return {std::move(guard), database_};
}

Storage::~Storage() { sqlite3_close(database_); }

} // namespace mfq::server
