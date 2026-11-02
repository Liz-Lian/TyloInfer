#include "sessions.h"
#include "http.h"
#include "failure.h"
#include "instance.h"
#include <charconv>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace mfq::server {
namespace {
using Json = nlohmann::json;
std::size_t page(const httplib::Request& request, const char* key, std::size_t fallback, std::size_t minimum, std::size_t maximum) {
    if (!request.has_param(key)) return fallback;
    const auto value = request.get_param_value(key);
    std::size_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result < minimum || result > maximum)
        throw ApiError(400, "invalid_request", std::string("invalid ") + key);
    return result;
}
std::string event_timestamp() {
    const auto now = std::time(nullptr);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream out; out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S+00:00"); return out.str();
}
struct ResponseStream {
    StreamSlot slot;
    std::shared_ptr<Sessions::Reply> reply;
    explicit ResponseStream(std::shared_ptr<std::atomic<std::size_t>> active) : slot(std::move(active)) {}
};
}
void register_sessions(httplib::Server& http, Sessions& sessions, std::shared_ptr<std::atomic<std::size_t>> active) {
    http.Post("/api/v1/sessions", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, sessions.create(parse_json(request)), 201);
    });
    http.Get("/api/v1/sessions", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, sessions.list(page(request, "limit", 50, 1, 200), page(request, "offset", 0, 0, INT64_MAX)));
    });
    http.Post("/api/v1/sessions/import", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, sessions.import_archive(parse_json(request)), 201);
    });
    http.Get("/api/v1/sessions/([0-9a-fA-F-]{36})", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, sessions.get(normalize_uuid(request.matches[1])));
    });
    http.Patch("/api/v1/sessions/([0-9a-fA-F-]{36})", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, sessions.update(normalize_uuid(request.matches[1]), parse_json(request)));
    });
    http.Delete("/api/v1/sessions/([0-9a-fA-F-]{36})", [&](const httplib::Request& request, httplib::Response& response) {
        sessions.erase(normalize_uuid(request.matches[1])); response.status = 204;
    });
    http.Get("/api/v1/sessions/([0-9a-fA-F-]{36})/messages", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, sessions.messages(normalize_uuid(request.matches[1])));
    });
    http.Post("/api/v1/sessions/([0-9a-fA-F-]{36})/messages", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, sessions.append(normalize_uuid(request.matches[1]), parse_json(request)), 201);
    });
    http.Post("/api/v1/sessions/([0-9a-fA-F-]{36})/fork", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, sessions.fork(normalize_uuid(request.matches[1]), parse_json(request)), 201);
    });
    http.Post("/api/v1/sessions/([0-9a-fA-F-]{36})/rewind", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, sessions.rewind(normalize_uuid(request.matches[1]), parse_json(request)));
    });
    http.Get("/api/v1/sessions/([0-9a-fA-F-]{36})/export", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, sessions.export_archive(normalize_uuid(request.matches[1])));
    });
    http.Get("/api/v1/sessions/([0-9a-fA-F-]{36})/responses", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, sessions.responses(normalize_uuid(request.matches[1]), page(request, "limit", 200, 1, 1000)));
    });
    http.Post("/api/v1/sessions/([0-9a-fA-F-]{36})/responses/cancel", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, sessions.cancel(normalize_uuid(request.matches[1])));
    });
    http.Post("/api/v1/sessions/([0-9a-fA-F-]{36})/responses", [&, active](const httplib::Request& request, httplib::Response& response) {
        const auto body = parse_json(request);
        auto stream = std::make_shared<ResponseStream>(active);
        const auto id = normalize_uuid(request.matches[1]);
        stream->reply = sessions.start(id, body);
        response.set_header("X-Request-ID", stream->reply->snapshot().at("request_id").get<std::string>());
        if (body.value("stream", true) == false) {
            while (stream->reply->next(request.is_connection_closed)) {}
            set_json(response, stream->reply->snapshot()); return;
        }
        response.set_header("Cache-Control", "no-cache"); response.set_header("X-Accel-Buffering", "no");
        response.set_chunked_content_provider("text/event-stream; charset=utf-8",
            [&, stream, id, disconnected = request.is_connection_closed](std::size_t, httplib::DataSink& sink) {
                std::size_t sequence = 0;
                bool connected = true;
                auto send = [&](Json payload) {
                    if (!connected) return;
                    const auto frame = Json{{"protocol_version", "1.0"}, {"session_id", id}, {"sequence", sequence},
                        {"timestamp", event_timestamp()}, {"payload", payload}};
                    const auto encoded = "event: " + payload.at("type").get<std::string>() + "\nid: " + std::to_string(sequence++) + "\ndata: " + frame.dump() + "\n\n";
                    if (disconnected() || !sink.write(encoded.data(), encoded.size())) { connected = false; stream->reply->cancel(); }
                };
                try {
                    const auto initial = sessions.get(id);
                    send({{"type", "session.state"}, {"state", initial.at("state")}, {"revision", initial.at("revision")}});
                    const auto resource = stream->reply->snapshot();
                    const auto response_id = resource.at("id");
                    if (resource.at("status") != "running") {
                        std::size_t tool_index = 0;
                        for (const auto& part : resource.at("output")) {
                            if (part.at("type") == "text" || part.at("type") == "reasoning")
                                send({{"type", part.at("type") == "text" ? "response.text.delta" : "response.reasoning.delta"}, {"response_id", response_id}, {"delta", part.at("text")}});
                            else if (part.at("type") == "tool_call")
                                send({{"type", "response.tool_call.delta"}, {"response_id", response_id}, {"index", tool_index++},
                                    {"call_id", part.at("call_id")}, {"name", part.at("name")}, {"arguments_delta", part.at("arguments").dump()}});
                        }
                    }
                    while (auto diffs = stream->reply->next(disconnected)) {
                        if (connected && disconnected()) { connected = false; stream->reply->cancel(); }
                        for (const auto& diff : *diffs) {
                            if (!diff.content_delta.empty()) send({{"type", "response.text.delta"}, {"response_id", response_id}, {"delta", diff.content_delta}});
                            if (!diff.reasoning_content_delta.empty()) send({{"type", "response.reasoning.delta"}, {"response_id", response_id}, {"delta", diff.reasoning_content_delta}});
                            if (diff.tool_call_index != std::string::npos)
                                send({{"type", "response.tool_call.delta"}, {"response_id", response_id}, {"index", diff.tool_call_index},
                                    {"call_id", diff.tool_call_delta.id}, {"name", diff.tool_call_delta.name}, {"arguments_delta", diff.tool_call_delta.arguments}});
                        }
                    }
                    const auto completed = stream->reply->snapshot();
                    const auto current = sessions.get(id);
                    if (completed.at("status") == "completed") send({{"type", "response.completed"}, {"response_id", response_id},
                        {"finish_reason", completed.at("finish_reason")}, {"usage", completed.at("usage")}, {"performance", completed.at("performance")}});
                    else if (completed.at("status") == "cancelled") send({{"type", "response.interrupted"}, {"response_id", response_id}, {"reason", "cancelled"}});
                    else send({{"type", "error"}, {"error", completed.at("error")}});
                    send({{"type", "session.state"}, {"state", current.at("state")}, {"revision", current.at("revision")}});
                } catch (...) {
                    const auto failure = describe_failure(std::current_exception());
                    send({{"type", "error"}, {"error", failure.detail}});
                }
                if (connected) sink.done();
                return false;
            }, [stream](bool) {});
    });
    http.Get("/api/v1/presets", [&](const httplib::Request&, httplib::Response& response) { set_json(response, sessions.presets()); });
    http.Post("/api/v1/presets", [&](const httplib::Request& request, httplib::Response& response) { set_json(response, sessions.put_preset(parse_json(request)), 201); });
    http.Put("/api/v1/presets/([0-9a-fA-F-]{36})", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, sessions.put_preset(parse_json(request), normalize_uuid(request.matches[1])));
    });
    http.Delete("/api/v1/presets/([0-9a-fA-F-]{36})", [&](const httplib::Request& request, httplib::Response& response) {
        sessions.erase_preset(normalize_uuid(request.matches[1])); response.status = 204;
    });
}
} // namespace mfq::server
