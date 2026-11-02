#include "jobs.h"
#include "jobs/jobs.h"
#include "http.h"
#include "failure.h"
#include "instance.h"
#include <charconv>
#include <thread>

namespace mfq::server {
namespace {
std::int64_t number(const std::string& value, std::int64_t fallback, std::int64_t maximum) {
    if (value.empty()) return fallback;
    std::int64_t result;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result < 0 || result > maximum)
        throw ApiError(400, "invalid_request", "invalid event cursor or limit");
    return result;
}
}
void register_jobs(httplib::Server& http, Jobs& jobs, std::shared_ptr<std::atomic<std::size_t>> streams) {
    http.Get("/api/v1/jobs", [&](const httplib::Request& request, httplib::Response& response) {
        for (const auto* key : {"limit", "offset", "status", "kind"})
            if (request.has_param(key) && request.get_param_value(key).empty()) throw ApiError(400, "invalid_request", "job filters cannot be empty");
        set_json(response, jobs.list(number(request.get_param_value("limit"), 100, 1000),
            number(request.get_param_value("offset"), 0, INT64_MAX), request.get_param_value("status"), request.get_param_value("kind")));
    });
    http.Delete("/api/v1/jobs/completed", [&](const httplib::Request&, httplib::Response& response) {
        jobs.clear(); response.status = 204;
    });
    http.Delete("/api/v1/jobs/([0-9a-f-]{36})", [&](const httplib::Request& request, httplib::Response& response) {
        jobs.erase(normalize_uuid(request.matches[1])); response.status = 204;
    });
    http.Get("/api/v1/jobs/([0-9a-f-]{36})", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, jobs.get(normalize_uuid(request.matches[1])));
    });
    http.Post("/api/v1/jobs/([0-9a-f-]{36})/cancel", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, jobs.cancel(normalize_uuid(request.matches[1])));
    });
    http.Get("/api/v1/jobs/([0-9a-f-]{36})/events", [&](const httplib::Request& request, httplib::Response& response) {
        const auto id = normalize_uuid(request.matches[1]);
        const auto limit = number(request.get_param_value("limit"), 256, 1000);
        auto after = number(request.get_param_value("after"), 0, INT64_MAX);
        if (request.get_param_value("tail") == "true")
            after = std::max<std::int64_t>(after, std::max<std::int64_t>(0, jobs.get(id)["last_sequence"].get<std::int64_t>() - std::min<std::int64_t>(limit, 256)));
        set_json(response, jobs.events(id, after, limit));
    });
    http.Get("/api/v1/jobs/([0-9a-f-]{36})/events/stream", [&, streams](const httplib::Request& request, httplib::Response& response) {
        const auto id = normalize_uuid(request.matches[1]);
        auto cursor = std::max(number(request.get_param_value("after"), 0, INT64_MAX),
            number(request.get_header_value("Last-Event-ID"), 0, INT64_MAX));
        (void)jobs.events(id, cursor);
        auto slot = std::make_shared<StreamSlot>(streams);
        const auto disconnected = request.is_connection_closed;
        response.set_header("Cache-Control", "no-cache");
        response.set_header("X-Accel-Buffering", "no");
        response.set_chunked_content_provider("text/event-stream; charset=utf-8",
            [&, id, cursor, slot, disconnected](std::size_t, httplib::DataSink& sink) mutable {
                try {
                    while (!disconnected()) {
                        const auto events = jobs.events(id, cursor).at("data");
                        for (const auto& event : events) {
                            const auto sequence = event.at("sequence").get<std::int64_t>();
                            const auto frame = "id: " + std::to_string(sequence) + "\ndata: " + event.dump() + "\n\n";
                            if (!sink.write(frame.data(), frame.size())) return false;
                            cursor = sequence;
                        }
                        const auto snapshot = jobs.get(id);
                        if (Jobs::terminal(snapshot) && cursor >= snapshot.at("last_sequence").get<std::int64_t>()) break;
                        constexpr char heartbeat[] = ": keepalive\n\n";
                        if (!sink.write(heartbeat, sizeof(heartbeat) - 1)) return false;
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                } catch (...) {
                    const auto frame = "event: error\ndata: " + nlohmann::json{{"error", describe_failure(std::current_exception()).detail}}.dump() + "\n\n";
                    sink.write(frame.data(), frame.size());
                }
                sink.done(); return false;
            }, [slot](bool) {});
    });
}
}
