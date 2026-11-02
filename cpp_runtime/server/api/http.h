#pragma once

#include "errors.h"
#include "httplib.h"
#include <algorithm>
#include <cctype>
#include <atomic>
#include <memory>
#include <nlohmann/json.hpp>

namespace mfq::server {
inline constexpr std::size_t http_workers = 4;
struct StreamSlot {
    std::shared_ptr<std::atomic<std::size_t>> active;
    explicit StreamSlot(std::shared_ptr<std::atomic<std::size_t>> count) : active(std::move(count)) {
        if (active->fetch_add(1) >= http_workers - 1) {
            --*active;
            throw ApiError(429, "stream_connections_full", "server streaming connections are full");
        }
    }
    ~StreamSlot() { --*active; }
};
inline nlohmann::json parse_json(const httplib::Request& request) {
    auto type = request.get_header_value("Content-Type");
    std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (type.substr(0, type.find(';')) != "application/json")
        throw ApiError(415, "unsupported_media_type", "Content-Type must be application/json");
    try { return nlohmann::json::parse(request.body); }
    catch (const nlohmann::json::exception&) { throw ApiError(400, "invalid_request", "invalid JSON request"); }
}
inline void set_json(httplib::Response& response, const nlohmann::json& body, int status = 200) {
    response.status = status;
    response.set_content(body.dump(), "application/json; charset=utf-8");
}
}
