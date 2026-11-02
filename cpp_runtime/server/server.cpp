#include "server.h"
#include "instance.h"
#include "api/auth.h"
#include "api/errors.h"
#include "api/http.h"
#include "api/inference.h"
#include "api/models.h"
#include "api/jobs.h"
#include "api/hub.h"
#include "hub/hub.h"
#include "api/sessions.h"
#include "api/failure.h"
#include "openapi.h"
#include "protocol.h"
#include "models/catalog.h"
#include "models/instances.h"
#include "models/errors.h"

#include "httplib.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <sstream>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <poll.h>
#include <unistd.h>
#endif

namespace mfq::server {
namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool loopback_host(const std::string& host) {
    in_addr ipv4{};
    if (inet_pton(AF_INET, host.c_str(), &ipv4) == 1) return (ntohl(ipv4.s_addr) >> 24) == 127;
    in6_addr ipv6{};
    if (inet_pton(AF_INET6, host.c_str(), &ipv6) != 1) throw std::invalid_argument("--host must be a numeric IP address");
    if (IN6_IS_ADDR_LOOPBACK(&ipv6)) return true;
    const auto* bytes = reinterpret_cast<const unsigned char*>(&ipv6);
    return std::all_of(bytes, bytes + 10, [](unsigned char c) { return c == 0; }) &&
        bytes[10] == 255 && bytes[11] == 255 && bytes[12] == 127;
}

bool cors_origin(const std::string& origin) {
    constexpr std::array<std::string_view, 7> allowed{
        "tauri://localhost", "http://tauri.localhost", "https://tauri.localhost",
        "http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:5174", "http://localhost:5174"};
    return std::find(allowed.begin(), allowed.end(), origin) != allowed.end();
}

void set_cors(const httplib::Request& request, httplib::Response& response) {
    const auto origin = request.get_header_value("Origin");
    if (cors_origin(origin)) {
        response.set_header("Access-Control-Allow-Origin", origin);
        response.set_header("Access-Control-Expose-Headers", "X-Request-ID");
    }
    if (!origin.empty()) response.set_header("Vary", "Origin");
}

void error(httplib::Response& response, int status, const char* code, const char* message) {
    set_json(response, {{"error", {{"code", code}, {"message", message}, {"retryable", status == 429 || status == 503}, {"details", json::object()}}}}, status);
}

void request_error(const httplib::Request& request, httplib::Response& response, std::exception_ptr exception) {
    const auto failure = describe_failure(exception);
    set_json(response, {{"error", failure.detail}}, failure.status);
    if (failure.status == 401) response.set_header("WWW-Authenticate", "Bearer");
    if (failure.detail.value("retryable", false) && (failure.status == 429 || failure.status == 503))
        response.set_header("Retry-After", "1");
    if (request.path == "/v1" || request.path.rfind("/v1/", 0) == 0)
        set_json(response, compat_error(failure, anthropic_request(request)), failure.status);
    set_cors(request, response);
}

bool within(const fs::path& root, const fs::path& path) {
    return std::mismatch(root.begin(), root.end(), path.begin(), path.end()).first == root.end();
}

bool reserved_path(const std::string& path) {
    constexpr std::array<std::string_view, 8> reserved{
        "api", "v1", "realtime", "health", "docs", "redoc", "assets", "static"};
    const auto first = std::string_view(path).substr(1, path.find('/', 1) - 1);
    return std::find(reserved.begin(), reserved.end(), first) != reserved.end();
}

bool invalid_path(const std::string& path) {
    if (path.find('\0') != std::string::npos || path.find('\\') != std::string::npos) return true;
    for (const auto& part : fs::u8path(path)) if (part == "..") return true;
    return false;
}

void static_response(const fs::path& root, const httplib::Request& request, httplib::Response& response) {
    if (invalid_path(request.path)) {
        error(response, 404, "not_found", "resource not found");
        return;
    }
    const auto first = std::string_view(request.path).substr(1, request.path.find('/', 1) - 1);
    if (reserved_path(request.path) && first != "assets" && first != "static") {
        error(response, 404, "not_found", "resource not found");
        return;
    }
    auto path = fs::weakly_canonical(root / fs::u8path(request.path).relative_path());
    if (fs::is_directory(path)) path = fs::weakly_canonical(path / "index.html");
    if (!within(root, path)) {
        error(response, 403, "forbidden", "resource is outside the web root");
        return;
    }
    if (!fs::is_regular_file(path)) {
        const auto accept = lowercase(request.get_header_value("Accept"));
        const auto destination = request.get_header_value("Sec-Fetch-Dest");
        auto normalized = request.path;
        while (!normalized.empty() && normalized.back() == '/') normalized.pop_back();
        if (reserved_path(request.path) || accept.find("text/html") == std::string::npos ||
            (destination != "" && destination != "document" && destination != "iframe") ||
            fs::u8path(normalized).has_extension()) {
            error(response, 404, "not_found", "resource not found");
            return;
        }
        path = fs::weakly_canonical(root / "index.html");
        if (!within(root, path)) {
            error(response, 403, "forbidden", "resource is outside the web root");
            return;
        }
    }
    response.set_file_content(path_utf8(path));
}

bool stdin_closed() {
    char buffer[1024];
#ifdef _WIN32
    const auto input = GetStdHandle(STD_INPUT_HANDLE);
    if (!input || input == INVALID_HANDLE_VALUE) return true;
    const auto kind = GetFileType(input);
    if (kind == FILE_TYPE_CHAR) return false;
    if (kind == FILE_TYPE_PIPE) {
        DWORD available = 0;
        if (!PeekNamedPipe(input, nullptr, 0, nullptr, &available, nullptr)) return true;
        if (!available) return false;
    }
    DWORD count = 0;
    return !ReadFile(input, buffer, sizeof(buffer), &count, nullptr) || count == 0;
#else
    pollfd input{STDIN_FILENO, POLLIN, 0};
    const auto ready = ::poll(&input, 1, 0);
    if (ready < 0) return errno != EINTR;
    if (!ready) return false;
    if (input.revents & (POLLERR | POLLNVAL)) return true;
    const auto count = ::read(STDIN_FILENO, buffer, sizeof(buffer));
    return count == 0 || (count < 0 && errno != EINTR && errno != EAGAIN);
#endif
}

} // namespace

int run(const Config& config, const std::atomic<bool>& stop_requested, EngineFactory factory) {
    const bool network = !loopback_host(config.host);
    if (config.api_key_env.empty() || !std::all_of(config.api_key_env.begin(), config.api_key_env.end(), [](unsigned char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
        }) || (config.api_key_env[0] >= '0' && config.api_key_env[0] <= '9'))
        throw std::invalid_argument("--api-key-env must be an environment variable name");
    fs::path web_root;
    if (!config.web_root.empty()) {
        web_root = resolved_path(config.web_root);
        const auto index = fs::weakly_canonical(web_root / "index.html");
        if (!fs::is_directory(web_root) || !fs::is_regular_file(index) || !within(web_root, index))
            throw std::invalid_argument("--web-root must contain an index.html inside that directory");
    }
    const auto data_dir = resolved_path(config.data_dir);
    InstanceLock instance(data_dir, config.instance_id, config.exit_on_stdin_close);
    const auto* root_key = std::getenv(config.api_key_env.c_str());
    auto database = config.database.empty() ? data_dir / "server.sqlite3" : resolved_path(config.database);
    if (config.database.empty() && !fs::exists(database) && fs::exists(data_dir / "tyloinfer-server.sqlite3"))
        database = data_dir / "tyloinfer-server.sqlite3";
    Storage storage(database);
    Authentication auth(storage, data_dir / "auth.json", root_key ? root_key : "", network);
    auto roots = config.model_dirs;
    if (roots.empty()) {
        if (const auto* configured = std::getenv("TYLOINFER_SERVER_MODEL_DIRS")) {
            std::istringstream supplied(configured);
            std::string root;
#ifdef _WIN32
            constexpr char separator = ';';
#else
            constexpr char separator = ':';
#endif
            while (std::getline(supplied, root, separator)) if (!root.empty()) roots.push_back(fs::u8path(root));
        }
    }
    if (roots.empty()) roots.push_back(data_dir / "models");
    ModelCatalog models(std::move(roots));
    Hub::Network hub_network;
    for (auto [key, value] : {std::pair{"HF_ENDPOINT", &hub_network.hf.url}, {"HF_TOKEN", &hub_network.hf.token},
            {"MODELSCOPE_ENDPOINT", &hub_network.ms.url}, {"MODELSCOPE_API_TOKEN", &hub_network.ms.token}})
        if (const auto* setting = std::getenv(key); setting && *setting) *value = setting;
    if (const auto* ca = std::getenv("SSL_CERT_FILE"); ca && *ca) hub_network.ca = fs::u8path(ca);
    Hub hub(data_dir, std::move(hub_network));
    ModelInstances instances(models, storage, std::move(factory));
    Sessions sessions(storage, instances);
    Jobs jobs(storage, [](std::exception_ptr exception) { return describe_failure(exception).detail; });
    auto streams = std::make_shared<std::atomic<std::size_t>>(0);
    httplib::Server http;
    http.set_socket_options([](socket_t socket) {
        // A public listener must fail on conflicts, not share traffic via SO_REUSEPORT.
#ifdef _WIN32
        constexpr int option = SO_EXCLUSIVEADDRUSE;
#else
        constexpr int option = SO_REUSEADDR;
#endif
        if (!httplib::set_socket_opt(socket, SOL_SOCKET, option, 1))
            throw std::runtime_error("cannot configure exclusive HTTP listener");
    });
    http.new_task_queue = [] { return new httplib::ThreadPool(http_workers, 0, 32); };
    http.set_payload_max_length(1024 * 1024);
    http.set_read_timeout(5);
    http.set_write_timeout(5);
    http.set_keep_alive_timeout(1);
    std::atomic<bool> draining{false};
    http.set_pre_routing_handler([&](const httplib::Request& request, httplib::Response& response) {
        const bool websocket = lowercase(request.get_header_value("Upgrade")) == "websocket";
        auto expected = request.get_header_value("X-TyloInfer-Instance-ID");
        if (websocket && expected.empty()) expected = request.get_param_value("server_instance");
        if (!expected.empty() && expected != instance.identity()["instance_id"].get<std::string>()) {
            request_error(request, response, std::make_exception_ptr(ApiError(409, "server_instance_changed", "The server instance changed. Reconnect and confirm the new server.")));
            return httplib::Server::HandlerResponse::Handled;
        }
        if (request.method == "OPTIONS" && request.has_header("Access-Control-Request-Method")) {
            constexpr const char* methods = "GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS";
            const auto method = request.get_header_value("Access-Control-Request-Method");
            bool permitted = cors_origin(request.get_header_value("Origin")) &&
                (method == "GET" || method == "HEAD" || method == "POST" || method == "PUT" ||
                 method == "PATCH" || method == "DELETE" || method == "OPTIONS");
            constexpr std::array<std::string_view, 9> headers{
                "accept", "authorization", "content-type", "x-content-sha256", "x-api-key", "anthropic-version", "anthropic-beta", "x-tyloinfer-instance-id", "last-event-id"};
            std::istringstream supplied(lowercase(request.get_header_value("Access-Control-Request-Headers")));
            std::string header;
            while (std::getline(supplied, header, ',')) {
                const auto begin = header.find_first_not_of(" \t"), end = header.find_last_not_of(" \t");
                header = begin == std::string::npos ? "" : header.substr(begin, end - begin + 1);
                if (std::find(headers.begin(), headers.end(), header) == headers.end()) permitted = false;
            }
            if (permitted) {
                response.status = 204;
                response.set_header("Access-Control-Allow-Methods", methods);
                response.set_header("Access-Control-Allow-Headers", request.get_header_value("Access-Control-Request-Headers"));
            } else error(response, 403, "cors_denied", "CORS request is not permitted");
            set_cors(request, response);
            return httplib::Server::HandlerResponse::Handled;
        }
        if (request.path.rfind("/api/", 0) == 0 || request.path.rfind("/v1", 0) == 0) {
            const auto authorization = request.get_header_value("Authorization");
            auto token = authorization.rfind("Bearer ", 0) == 0 ? authorization.substr(7) : "";
            if (token.empty() && websocket) token = request.get_param_value("access_token");
            if (token.empty() && anthropic_request(request)) token = request.get_header_value("X-Api-Key");
            try { auth.require(token, required_scope(request.method, request.path)); }
            catch (...) {
                request_error(request, response, std::current_exception());
                return httplib::Server::HandlerResponse::Handled;
            }
        }
        const bool read = request.method == "GET" || request.method == "HEAD" || request.method == "OPTIONS";
        const bool cancel = request.method == "POST" && request.path.size() >= 7 &&
            request.path.compare(request.path.size() - 7, 7, "/cancel") == 0;
        if ((draining || stop_requested.load(std::memory_order_relaxed)) && !read && !cancel) {
            request_error(request, response, std::make_exception_ptr(ApiError(503, "server_stopping", "server is stopping")));
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    http.set_post_routing_handler([](const httplib::Request& request, httplib::Response& response) {
        set_cors(request, response);
        response.set_header("X-Content-Type-Options", "nosniff");
        response.set_header("X-Frame-Options", "DENY");
        response.set_header("Referrer-Policy", "no-referrer");
        response.set_header("Content-Security-Policy", "default-src 'self'; connect-src 'self' https://api.github.com ws: wss:; "
            "img-src 'self' data: blob:; media-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; font-src 'self' data:");
    });
    http.set_error_handler([](const httplib::Request& request, httplib::Response& response) {
        if (response.body.empty()) request_error(request, response,
            std::make_exception_ptr(ApiError(response.status, "http_error", "request failed")));
    });
    http.set_exception_handler(request_error);
    http.Get("/openapi.json", [](const httplib::Request&, httplib::Response& response) {
        response.set_content(openapi_document, "application/json; charset=utf-8");
    });
    http.Get("/health", [](const httplib::Request&, httplib::Response& response) {
        set_json(response, {{"status", "ok"}, {"service", "tyloinfer-server"}, {"protocol_version", "1.0"}});
    });
    http.Get("/health/identity", [&](const httplib::Request&, httplib::Response& response) {
        set_json(response, instance.identity());
    });
    register_auth(http, auth);
    register_models(http, models, instances, jobs);
    register_jobs(http, jobs, streams);
    register_hub(http, hub, models, jobs);
    register_inference(http, instances, streams);
    register_sessions(http, sessions, streams);
    if (!web_root.empty()) http.Get("/.*", [&](const httplib::Request& request, httplib::Response& response) {
        static_response(web_root, request, response);
    });
    const int port = config.port == 0 ? http.bind_to_any_port(config.host) :
        (http.bind_to_port(config.host, config.port) ? config.port : -1);
    if (port < 0) throw std::runtime_error("cannot bind listener on " + config.host + ":" + std::to_string(config.port));
    http.set_start_handler([&] {
        std::cout << json{{"event", "listening"}, {"host", config.host}, {"port", port},
            {"instance_id", instance.identity()["instance_id"]}}.dump() << std::endl;
    });
    std::atomic<bool> finished{false};
    std::atomic<bool> shutdown_failed{false};
    auto shutdown = [&] {
        if (draining.exchange(true)) return;
        jobs.request_stop();
        try { sessions.request_stop(); } catch (const std::exception& failure) {
            shutdown_failed = true;
            std::cerr << json{{"event", "session_stop_failed"}, {"detail", failure.what()}}.dump() << '\n';
        }
        try { instances.request_stop(); } catch (const std::exception& failure) {
            shutdown_failed = true;
            std::cerr << json{{"event", "model_stop_failed"}, {"detail", failure.what()}}.dump() << '\n';
        }
        try { jobs.shutdown(); } catch (const std::exception& failure) {
            shutdown_failed = true;
            std::cerr << json{{"event", "operation_shutdown_failed"}, {"detail", failure.what()}}.dump() << '\n';
        }
        try { sessions.shutdown(); } catch (const std::exception& failure) {
            shutdown_failed = true;
            std::cerr << json{{"event", "session_shutdown_failed"}, {"detail", failure.what()}}.dump() << '\n';
        }
        try { instances.shutdown(); } catch (const std::exception& failure) {
            shutdown_failed = true;
            std::cerr << json{{"event", "model_shutdown_failed"}, {"detail", failure.what()}}.dump() << '\n';
        }
        http.stop();
    };
    std::thread lifecycle([&] {
        while (!finished.load(std::memory_order_relaxed)) {
            if (http.is_running() && (stop_requested.load(std::memory_order_relaxed) ||
                    (config.exit_on_stdin_close && stdin_closed()))) {
                shutdown();
                return;
            }
            try { instances.reap_idle(); } catch (const std::exception& failure) {
                std::cerr << json{{"event", "idle_unload_failed"}, {"detail", failure.what()}}.dump() << '\n';
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
    });
    bool successful;
    try {
        successful = http.listen_after_bind();
    } catch (...) {
        finished.store(true, std::memory_order_relaxed);
        lifecycle.join();
        shutdown();
        throw;
    }
    finished.store(true, std::memory_order_relaxed);
    lifecycle.join();
    shutdown();
    if (!successful) throw std::runtime_error("HTTP listener failed");
    std::cerr << "TyloInfer server stopped\n";
    return shutdown_failed ? 1 : 0;
}

} // namespace mfq::server
