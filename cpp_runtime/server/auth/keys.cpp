#include "auth.h"
#include "storage/sqlite.h"

namespace mfq::server {
namespace {
using json = nlohmann::json;
using sqlite::Statement;
constexpr const char* key_columns = "id,name,prefix,scopes_json,role,expires_at,revoked_at,created_at,last_used_at";

json key_row(const Statement& query) {
    return {{"id", query.column(0)}, {"name", query.column(1)}, {"prefix", query.column(2)},
        {"scopes", json::parse(query.column(3).get<std::string>())}, {"role", query.column(4)},
        {"expires_at", query.column(5)}, {"revoked_at", query.column(6)},
        {"created_at", query.column(7)}, {"last_used_at", query.column(8)}};
}

json get_key(sqlite3* database, const std::string& id) {
    const auto sql = "SELECT " + std::string(key_columns) + " FROM api_keys WHERE id=?";
    Statement query(database, sql.c_str());
    query.bind(1, id);
    if (!query.step()) throw AuthError(AuthError::not_found, "api_key_not_found", "API key not found");
    return key_row(query);
}

}

json Authentication::store_key(const json& request, const std::string& id, const std::string& hash, const std::string& prefix) try {
    auto access = storage_.access();
    auto* database = access.database;
    sqlite::Transaction transaction(database);
    {
        Statement insert(database, "INSERT INTO api_keys(id,name,key_hash,prefix,scopes_json,role,expires_at,created_at) "
            "VALUES(?,?,?,?,?,?,?,strftime('%Y-%m-%dT%H:%M:%f+00:00','now'))");
        insert.bind(1, id);
        insert.bind(2, request.at("name"));
        insert.bind(3, hash);
        insert.bind(4, prefix);
        insert.bind(5, request.at("scopes").dump());
        insert.bind(6, request.at("role"));
        insert.bind(7, request.at("expires_at"));
        insert.step();
    }
    auto result = get_key(database, id);
    transaction.commit();
    return result;
} catch (const sqlite::Error& error) {
    if ((error.code & 255) == SQLITE_CONSTRAINT)
        throw AuthError(AuthError::conflict, "api_key_conflict", "API key name or credential already exists");
    throw;
}

json Authentication::list_keys() {
    auto access = storage_.access();
    auto* database = access.database;
    const auto sql = "SELECT " + std::string(key_columns) + " FROM api_keys ORDER BY created_at DESC,id";
    Statement query(database, sql.c_str());
    json keys = json::array();
    while (query.step()) keys.push_back(key_row(query));
    return {{"data", std::move(keys)}};
}

std::optional<json> Authentication::authenticate_key(const std::string& hash) {
    auto access = storage_.access();
    auto* database = access.database;
    Statement update(database, "UPDATE api_keys SET last_used_at=strftime('%Y-%m-%dT%H:%M:%f+00:00','now') "
        "WHERE key_hash=? AND revoked_at IS NULL AND (expires_at IS NULL OR julianday(expires_at)>julianday('now'))");
    update.bind(1, hash);
    update.step();
    if (!sqlite3_changes(database)) return std::nullopt;
    const auto sql = "SELECT " + std::string(key_columns) + " FROM api_keys WHERE key_hash=?";
    Statement query(database, sql.c_str());
    query.bind(1, hash);
    if (!query.step()) return std::nullopt;
    return key_row(query);
}

json Authentication::revoke_key(const std::string& id) {
    auto access = storage_.access();
    auto* database = access.database;
    sqlite::Transaction transaction(database);
    Statement update(database, "UPDATE api_keys SET revoked_at=strftime('%Y-%m-%dT%H:%M:%f+00:00','now') WHERE id=?");
    update.bind(1, id);
    update.step();
    if (!sqlite3_changes(database)) throw AuthError(AuthError::not_found, "api_key_not_found", "API key not found");
    auto result = get_key(database, id);
    transaction.commit();
    return result;
}

json Authentication::replace_key(const std::string& id, const std::string& hash, const std::string& prefix) try {
    auto access = storage_.access();
    auto* database = access.database;
    sqlite::Transaction transaction(database);
    Statement update(database, "UPDATE api_keys SET key_hash=?,prefix=?,revoked_at=NULL,last_used_at=NULL WHERE id=?");
    update.bind(1, hash);
    update.bind(2, prefix);
    update.bind(3, id);
    update.step();
    if (!sqlite3_changes(database)) throw AuthError(AuthError::not_found, "api_key_not_found", "API key not found");
    auto result = get_key(database, id);
    transaction.commit();
    return result;
} catch (const sqlite::Error& error) {
    if ((error.code & 255) == SQLITE_CONSTRAINT)
        throw AuthError(AuthError::conflict, "api_key_conflict", "API key name or credential already exists");
    throw;
}

}
