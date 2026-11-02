#include "auth.h"
#include "http.h"
#include "instance.h"

namespace mfq::server {
Scope required_scope(const std::string& method, const std::string& path) {
    const auto starts = [&](const char* prefix) { return path.rfind(prefix, 0) == 0; };
    const bool read = method == "GET" || method == "HEAD";
    if (path == "/api/v1/hub/download-directory") return Scope::models;
    if (starts("/api/v1/runtime/cache/entries") || path == "/api/v1/runtime/cache/purge") return Scope::admin;
    if (!read && (path == "/api/v1/runtime/listener" || path == "/api/v1/runtime/memory-policy" ||
        path == "/api/v1/runtime/model-aliases" || path == "/api/v1/runtime/context-policy")) return Scope::admin;
    if (starts("/api/v1/auth/") || starts("/api/v1/cluster/")) return Scope::admin;
    if (starts("/api/v1/models/directories")) return Scope::models;
    if (starts("/api/v1/models/") && path.size() >= 10 && path.compare(path.size() - 10, 10, "/directory") == 0) return Scope::models;
    if (starts("/api/v1/models") || starts("/api/v1/runtime") || starts("/api/v1/hub"))
        return read ? Scope::inference : Scope::models;
    for (const auto* prefix : {"/api/v1/jobs", "/api/v1/artifacts", "/api/v1/datasets", "/api/v1/evaluations", "/api/v1/quantization"})
        if (starts(prefix)) return Scope::jobs;
    return Scope::inference;
}

void register_auth(httplib::Server& http, Authentication& auth) {
    http.Get("/api/v1/auth/policy", [&](const httplib::Request&, httplib::Response& response) {
        set_json(response, auth.status());
    });
    http.Put("/api/v1/auth/policy", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, auth.configure(parse_json(request)));
    });
    http.Get("/api/v1/auth/keys", [&](const httplib::Request&, httplib::Response& response) {
        set_json(response, auth.list_keys());
    });
    http.Post("/api/v1/auth/keys", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, auth.create_key(parse_json(request)), 201);
    });
    http.Post("/api/v1/auth/keys/([^/]+)/revoke", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, auth.revoke_key(normalize_uuid(request.matches[1])));
    });
    http.Post("/api/v1/auth/keys/([^/]+)/rotate", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, auth.rotate_key(request.matches[1]));
    });
}
}
