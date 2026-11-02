#pragma once

#include "storage/storage.h"
#include <optional>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace mfq::server {

enum class Scope { inference, models, jobs, admin };
struct AuthError : std::runtime_error {
    enum Kind { invalid, not_found, conflict, unauthorized, forbidden, save_failed } kind;
    std::string code;
    AuthError(Kind kind, const char* code, const std::string& message)
        : std::runtime_error(message), kind(kind), code(code) {}
};

class Authentication {
public:
    Authentication(Storage& storage, std::filesystem::path policy_path, const std::string& root_key, bool network);
    nlohmann::json status();
    nlohmann::json configure(const nlohmann::json& request);
    void require(const std::string& token, Scope scope);
    nlohmann::json create_key(nlohmann::json request);
    nlohmann::json rotate_key(const std::string& id);
    nlohmann::json list_keys();
    nlohmann::json revoke_key(const std::string& id);

private:
    nlohmann::json store_key(const nlohmann::json& request, const std::string& id,
        const std::string& hash, const std::string& prefix);
    nlohmann::json replace_key(const std::string& id, const std::string& hash, const std::string& prefix);
    std::optional<nlohmann::json> authenticate_key(const std::string& hash);
    Storage& storage_;
    std::filesystem::path path_;
    std::mutex mutex_;
    bool enabled_ = false;
    bool network_;
    std::string key_hash_;
};

} // namespace mfq::server
