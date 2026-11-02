#pragma once

#include <sqlite3.h>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace mfq::server::sqlite {
using json = nlohmann::json;

struct Error : std::runtime_error {
    int code;
    Error(sqlite3* database, int result)
        : std::runtime_error(std::string("SQLite: ") + sqlite3_errmsg(database)), code(result) {}
};

inline void check(sqlite3* database, int result) {
    if (result != SQLITE_OK && result != SQLITE_ROW && result != SQLITE_DONE) throw Error(database, result);
}

inline void execute(sqlite3* database, const char* sql) {
    check(database, sqlite3_exec(database, sql, nullptr, nullptr, nullptr));
}

class Statement {
public:
    Statement(sqlite3* database, const char* sql) : database_(database) {
        check(database_, sqlite3_prepare_v2(database_, sql, -1, &statement_, nullptr));
    }
    ~Statement() { sqlite3_finalize(statement_); }
    Statement(const Statement&) = delete;
    Statement& operator=(const Statement&) = delete;
    void bind(int index, const json& value) {
        if (value.is_null()) check(database_, sqlite3_bind_null(statement_, index));
        else {
            const auto text = value.get<std::string>();
            check(database_, sqlite3_bind_text(statement_, index, text.data(), static_cast<int>(text.size()), SQLITE_TRANSIENT));
        }
    }
    bool step() {
        const auto result = sqlite3_step(statement_);
        check(database_, result);
        return result == SQLITE_ROW;
    }
    json column(int index) const {
        const auto* text = sqlite3_column_text(statement_, index);
        return text ? json(std::string(reinterpret_cast<const char*>(text), sqlite3_column_bytes(statement_, index))) : json(nullptr);
    }

private:
    sqlite3* database_;
    sqlite3_stmt* statement_ = nullptr;
};

class Transaction {
public:
    explicit Transaction(sqlite3* database) : database_(database) { execute(database_, "BEGIN IMMEDIATE"); }
    ~Transaction() { if (active_) sqlite3_exec(database_, "ROLLBACK", nullptr, nullptr, nullptr); }
    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;
    void commit() { execute(database_, "COMMIT"); active_ = false; }
private:
    sqlite3* database_;
    bool active_ = true;
};
}
