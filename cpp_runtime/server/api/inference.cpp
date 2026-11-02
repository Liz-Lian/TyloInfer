#include "inference.h"
#include "anthropic.h"
#include "failure.h"
#include "http.h"
#include "instance.h"
#include "models/instances.h"
#include "protocol.h"
#include <charconv>
#include <map>

namespace mfq::server {
namespace {
using json = nlohmann::json;
struct Generation {
    StreamSlot slot;
    std::shared_ptr<mfq::protocol::CompletionStream> completion;
    explicit Generation(std::shared_ptr<std::atomic<std::size_t>> count) : slot(std::move(count)) {}
};

json chunk(const std::string& id, std::int64_t created, const std::string& model, json delta, json finish = nullptr) {
    return {{"id", "chatcmpl-" + id}, {"object", "chat.completion.chunk"}, {"created", created}, {"model", model},
        {"choices", json::array({{{"index", 0}, {"delta", std::move(delta)}, {"finish_reason", std::move(finish)}}})}};
}
bool sse(httplib::DataSink& sink, const json& value) {
    const auto text = "data: " + value.dump() + "\n\n";
    return sink.write(text.data(), text.size());
}
}

void register_inference(httplib::Server& http, ModelInstances& models, std::shared_ptr<std::atomic<std::size_t>> active) {
    register_anthropic(http, models, active);
    http.Get("/v1", [](const httplib::Request& request, httplib::Response& response) {
        set_json(response, {{"status", "ok"}, {"service", "tyloinfer-server"},
            {"endpoints", {"/v1/models", anthropic_request(request) ? "/v1/messages" : "/v1/chat/completions"}}});
    });
    http.Get("/v1/models", [&](const httplib::Request& request, httplib::Response& response) {
        auto data = json::array();
        const bool anthropic = anthropic_request(request);
        for (const auto& name : models.names()) {
            if (anthropic) data.push_back({{"id", name}, {"type", "model"}, {"display_name", name}, {"created_at", "1970-01-01T00:00:00Z"}});
            else data.push_back({{"id", name}, {"object", "model"}, {"created", 0}, {"owned_by", "mfq"}});
        }
        if (anthropic) {
            unsigned limit = 20;
            if (request.has_param("limit")) {
                const auto value = request.get_param_value("limit");
                const auto parsed = std::from_chars(value.data(), value.data() + value.size(), limit);
                if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !limit || limit > 1000)
                    throw ApiError(400, "invalid_request", "limit must be between 1 and 1000");
            }
            std::size_t begin = 0, end = data.size();
            for (const auto* cursor : {"after_id", "before_id"}) if (request.has_param(cursor)) {
                const auto id = request.get_param_value(cursor);
                const auto found = std::find_if(data.begin(), data.end(), [&](const json& value) { return value.at("id") == id; });
                if (found == data.end()) throw ApiError(400, "invalid_request", "unknown model cursor");
                const auto index = static_cast<std::size_t>(found - data.begin());
                if (std::string_view(cursor) == "after_id") begin = index + 1;
                else end = index;
            }
            auto page = json::array();
            for (auto i = begin; i < end && page.size() < limit; ++i) page.push_back(data[i]);
            set_json(response, {{"data", page}, {"has_more", begin + page.size() < end},
                {"first_id", page.empty() ? json(nullptr) : page.front().at("id")}, {"last_id", page.empty() ? json(nullptr) : page.back().at("id")}});
            return;
        }
        set_json(response, {{"object", "list"}, {"data", std::move(data)}});
    });
    http.Get("/v1/models/(.+)", [&](const httplib::Request& request, httplib::Response& response) {
        const std::string model = request.matches[1];
        const auto names = models.names();
        if (std::find(names.begin(), names.end(), model) == names.end()) models.get(model);
        set_json(response, {{"id", model}, {"type", "model"}, {"display_name", model}, {"created_at", "1970-01-01T00:00:00Z"}});
    });
    http.Post("/api/v1/inference/requests/([0-9a-f-]{36})/cancel", [&](const httplib::Request& request, httplib::Response& response) {
        const auto body = parse_json(request);
        if (!body.is_object() || body.size() != 1 || !body.contains("instance_id") || !body["instance_id"].is_string())
            throw ApiError(400, "invalid_request", "cancel requires an instance_id");
        const auto instance = models.get(normalize_uuid(body["instance_id"].get<std::string>()));
        set_json(response, {{"cancel_requested", instance->cancel(normalize_uuid(request.matches[1]))}});
    });
    http.Post("/v1/chat/completions", [&, active](const httplib::Request& request, httplib::Response& response) {
        const auto body = parse_json(request);
        if (!body.is_object() || !body.contains("model") || !body["model"].is_string())
            throw ApiError(400, "invalid_request", "model must select a registered model or loaded instance");
        const auto model = body["model"].get<std::string>();
        MfqSamplingParams validation; validation.temperature = 0; validation.top_p = 1; validation.top_k = 1;
        mfq::protocol::parse_input(body, true, validation);
        auto generation = std::make_shared<Generation>(active);
        auto instance = models.acquire(model, request.is_connection_closed);
        auto input = mfq::protocol::parse_input(body, true, instance->sampling());
        const bool stream = input.stream, include_usage = input.include_usage;
        const auto id = normalize_uuid({});
        const auto created = mfq::protocol::unix_time_seconds();
        auto outbox = instance->submit({id, std::move(input)});
        auto completion = generation->completion = std::make_shared<mfq::protocol::CompletionStream>(
            std::move(outbox), stream, [instance, id] { instance->cancel(id); });
        response.set_header("X-Request-ID", id);
        auto disconnected = request.is_connection_closed;
        if (!stream) {
            while (completion->next()) if (disconnected()) { completion->cancel(); return; }
            const auto& result = completion->result;
            if (result.cancelled) throw ApiError(409, "request_cancelled", "native generation was cancelled");
            json message{{"role", "assistant"}, {"content", result.text}};
            if (!result.reasoning_text.empty()) message["reasoning_content"] = result.reasoning_text;
            if (!result.tool_calls.empty()) {
                message["tool_calls"] = json::array();
                for (const auto& call : result.tool_calls)
                    message["tool_calls"].push_back({{"id", call.id}, {"type", "function"}, {"function", {{"name", call.name}, {"arguments", call.arguments}}}});
            }
            set_json(response, {{"id", "chatcmpl-" + id}, {"object", "chat.completion"}, {"created", created}, {"model", model},
                {"choices", json::array({{{"index", 0}, {"message", std::move(message)}, {"finish_reason", result.finish_reason}}})},
                {"usage", mfq::protocol::usage_json(completion->prompt_tokens, result.completion_tokens)}});
            return;
        }
        response.set_header("Cache-Control", "no-cache");
        response.set_header("X-Accel-Buffering", "no");
        response.set_chunked_content_provider("text/event-stream; charset=utf-8",
            [generation, id, created, model, include_usage, disconnected](std::size_t, httplib::DataSink& sink) {
                const auto& completion = generation->completion;
                try {
                    std::map<std::size_t, common_chat_tool_call> identities;
                    /** Convert complete native identities to append-only OpenAI field fragments. */
                    const auto identity_delta = [](std::string& previous, std::string& value) {
                        if (value.empty()) return;
                        if (value.compare(0, previous.size(), previous) != 0)
                            throw ApiError(502, "invalid_tool_output", "native tool identity changed after it was streamed");
                        auto complete = value;
                        value.erase(0, previous.size());
                        previous = std::move(complete);
                    };
                    if (!sse(sink, chunk(id, created, model, {{"role", "assistant"}}))) return false;
                    while (auto diffs = completion->next()) {
                        if (disconnected()) { completion->cancel(); return false; }
                        for (auto& diff : *diffs) {
                            if (diff.tool_call_index != std::string::npos) {
                                auto& identity = identities[diff.tool_call_index];
                                identity_delta(identity.id, diff.tool_call_delta.id);
                                identity_delta(identity.name, diff.tool_call_delta.name);
                            }
                            auto delta = mfq::protocol::chat_diff_json(diff);
                            if (!delta.empty() && !sse(sink, chunk(id, created, model, std::move(delta)))) {
                                completion->cancel(); return false;
                            }
                        }
                    }
                    if (completion->result.cancelled) throw ApiError(409, "request_cancelled", "native generation was cancelled");
                    if (!sse(sink, chunk(id, created, model, json::object(), completion->result.finish_reason))) return false;
                    if (include_usage) {
                        auto usage = chunk(id, created, model, json::object());
                        usage["choices"] = json::array();
                        usage["usage"] = mfq::protocol::usage_json(completion->prompt_tokens, completion->result.completion_tokens);
                        if (!sse(sink, usage)) return false;
                    }
                    constexpr char done[] = "data: [DONE]\n\n";
                    if (!sink.write(done, sizeof(done) - 1)) return false;
                } catch (...) {
                    sse(sink, compat_error(describe_failure(std::current_exception()), false));
                }
                sink.done();
                return false;
            }, [generation](bool) {});
    });
}
}
