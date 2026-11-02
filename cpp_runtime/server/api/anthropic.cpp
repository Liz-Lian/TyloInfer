#include "anthropic.h"
#include "failure.h"
#include "http.h"
#include "instance.h"
#include "models/instances.h"

#include <map>
#include <regex>
#include <set>

namespace mfq::server {
namespace {
using Json = nlohmann::json;
using ProtocolError = mfq::protocol::ApiError;

void fields(const Json& value, std::initializer_list<const char*> names) {
    if (!value.is_object()) throw ProtocolError(400, "invalid_request_error", "expected an object");
    for (const auto& item : value.items())
        if (std::find(names.begin(), names.end(), item.key()) == names.end())
            throw ProtocolError(400, "unsupported_parameter", "unsupported field: " + item.key(), item.key());
}
std::string text(const Json& value, const char* field, bool empty = false) {
    if (!value.is_string() || (!empty && value.get_ref<const std::string&>().empty()))
        throw ProtocolError(400, "invalid_request_error", std::string(field) + " must be a string" + (empty ? "" : " and nonempty"), field);
    return value.get<std::string>();
}
bool boolean(const Json& body, const char* field, bool fallback = false) {
    if (!body.contains(field)) return fallback;
    if (!body[field].is_boolean()) throw ProtocolError(400, "invalid_request_error", std::string(field) + " must be boolean", field);
    return body[field].get<bool>();
}
Json blocks(const Json& value, bool empty = false) {
    if (value.is_string()) return Json::array({{{"type", "text"}, {"text", value}}});
    if (!value.is_array() || (!empty && value.empty())) throw ProtocolError(400, "invalid_request_error", "content must be text or an array of blocks");
    return value;
}
Json content(const Json& block) {
    if (!block.is_object()) throw ProtocolError(400, "invalid_request_error", "content blocks must be objects");
    const auto kind = text(block.value("type", Json()), "content.type");
    if (kind == "text") {
        fields(block, {"type", "text"});
        return {{"type", "text"}, {"text", text(block.value("text", Json()), "text", true)}};
    }
    if (kind != "image") throw ProtocolError(400, "unsupported_parameter", "unsupported content block: " + kind);
    fields(block, {"type", "source"});
    const auto source = block.value("source", Json());
    fields(source, {"type", "url", "media_type", "data"});
    std::string url;
    if (source.value("type", Json()) == "url") {
        fields(source, {"type", "url"}); url = text(source.value("url", Json()), "image.source.url");
        if (url.rfind("http://", 0) != 0 && url.rfind("https://", 0) != 0)
            throw ProtocolError(400, "invalid_request_error", "image URLs must use HTTP or HTTPS");
    } else if (source.value("type", Json()) == "base64") {
        fields(source, {"type", "media_type", "data"});
        const auto mime = text(source.value("media_type", Json()), "image.source.media_type");
        if (mime != "image/jpeg" && mime != "image/png" && mime != "image/gif" && mime != "image/webp")
            throw ProtocolError(400, "invalid_request_error", "unsupported image media type");
        url = "data:" + mime + ";base64," + text(source.value("data", Json()), "image.source.data");
    } else throw ProtocolError(400, "invalid_request_error", "image source must be URL or base64");
    return {{"type", "image_url"}, {"image_url", {{"url", std::move(url)}}}};
}

engine::InferenceInput parse_messages(const Json& body, const MfqSamplingParams& defaults) {
    fields(body, {"model", "messages", "max_tokens", "system", "stream", "temperature", "top_p", "top_k", "tools",
        "tool_choice", "thinking", "metadata", "stop_sequences", "enable_mtp", "enable_thinking", "seed", "mfq_session_id"});
    text(body.value("model", Json()), "model");
    const auto maximum = mfq::protocol::integer_field(body, "max_tokens", 0);
    if (maximum < 1 || maximum > INT32_MAX) throw ProtocolError(400, "invalid_request_error", "max_tokens must be a positive 32-bit integer", "max_tokens");
    Json converted{{"max_tokens", maximum}, {"stream", boolean(body, "stream")}};
    for (const auto* field : {"temperature", "top_p", "top_k", "seed", "mfq_session_id", "enable_mtp", "enable_thinking"})
        if (body.contains(field)) converted[field] = body[field];
    if (body.contains("temperature")) {
        if (!body["temperature"].is_number()) throw ProtocolError(400, "invalid_request_error", "temperature must be numeric", "temperature");
        const auto temperature = mfq::protocol::number_field(body, "temperature", 0);
        if (temperature < 0 || temperature > 1) throw ProtocolError(400, "invalid_request_error", "temperature must be in [0, 1]", "temperature");
    }
    if (body.contains("top_k")) {
        const auto value = mfq::protocol::integer_field(body, "top_k", -1);
        if (value < 0 || value > 1024) throw ProtocolError(400, "invalid_request_error", "top_k must be in [0, 1024]", "top_k");
    }
    if (body.contains("top_p") && !body["top_p"].is_number()) throw ProtocolError(400, "invalid_request_error", "top_p must be numeric", "top_p");
    if (body.contains("seed") && mfq::protocol::integer_field(body, "seed", 0) < 0)
        throw ProtocolError(400, "invalid_request_error", "seed must be nonnegative", "seed");
    for (const auto* field : {"enable_mtp", "enable_thinking"}) if (body.contains(field)) boolean(body, field);
    if (body.contains("metadata")) {
        fields(body["metadata"], {"user_id"});
        if (body["metadata"].contains("user_id") && !body["metadata"]["user_id"].is_null()) text(body["metadata"]["user_id"], "metadata.user_id");
    }
    if (body.contains("stop_sequences")) {
        const auto& stops = body["stop_sequences"];
        if (!stops.is_array() || stops.size() > 16) throw ProtocolError(400, "invalid_request_error", "stop_sequences must be an array of at most 16 strings");
        for (const auto& stop : stops) text(stop, "stop_sequences");
        if (!stops.empty()) throw ProtocolError(501, "unsupported_parameter", "native terminal events do not expose the matched stop sequence");
    }
    if (body.contains("thinking")) {
        const auto& thinking = body["thinking"]; fields(thinking, {"type", "budget_tokens"});
        const auto kind = text(thinking.value("type", Json()), "thinking.type");
        if (kind == "enabled") {
            const auto budget = mfq::protocol::integer_field(thinking, "budget_tokens", 0);
            if (budget < 1024 || budget >= maximum) throw ProtocolError(400, "invalid_request_error", "thinking budget must be at least 1024 and below max_tokens");
            throw ProtocolError(501, "unsupported_parameter", "native thinking token budgets are not available");
        }
        if ((kind != "disabled" && kind != "adaptive") || thinking.contains("budget_tokens"))
            throw ProtocolError(400, "invalid_request_error", "thinking must select adaptive or disabled without a budget");
        if (converted.contains("enable_thinking") && converted["enable_thinking"] != (kind == "adaptive"))
            throw ProtocolError(400, "invalid_request_error", "conflicting thinking settings");
        converted["enable_thinking"] = kind == "adaptive";
    }
    auto messages = Json::array();
    if (body.contains("system")) {
        std::string system;
        for (const auto& block : blocks(body["system"])) {
            fields(block, {"type", "text"});
            if (block.value("type", Json()) != "text") throw ProtocolError(400, "invalid_request_error", "system accepts text only");
            if (!system.empty()) system += '\n';
            system += text(block.value("text", Json()), "system.text", true);
        }
        messages.push_back({{"role", "system"}, {"content", std::move(system)}});
    }
    const auto raw = body.value("messages", Json());
    if (!raw.is_array() || raw.empty()) throw ProtocolError(400, "invalid_request_error", "messages must be a nonempty array");
    for (const auto& message : raw) {
        fields(message, {"role", "content"});
        const auto role = text(message.value("role", Json()), "message.role");
        if (role != "user" && role != "assistant") throw ProtocolError(400, "invalid_request_error", "message role must be user or assistant");
        auto parts = Json::array(), calls = Json::array(); std::string reasoning;
        const auto flush = [&] {
            if (parts.empty() && calls.empty() && reasoning.empty()) return;
            Json output{{"role", role}, {"content", parts}};
            if (!calls.empty()) output["tool_calls"] = calls;
            if (!reasoning.empty()) output["reasoning_content"] = reasoning;
            messages.push_back(std::move(output)); parts.clear(); calls.clear(); reasoning.clear();
        };
        for (const auto& block : blocks(message.value("content", Json()))) {
            if (!block.is_object()) throw ProtocolError(400, "invalid_request_error", "content blocks must be objects");
            const auto kind = text(block.value("type", Json()), "content.type");
            if (kind == "tool_use" && role == "assistant") {
                fields(block, {"type", "id", "name", "input"});
                if (!block.contains("input") || !block["input"].is_object()) throw ProtocolError(400, "invalid_request_error", "tool input must be an object");
                calls.push_back({{"id", text(block.value("id", Json()), "tool_use.id")}, {"type", "function"},
                    {"function", {{"name", text(block.value("name", Json()), "tool_use.name")}, {"arguments", block["input"].dump()}}}});
            } else if (kind == "tool_result" && role == "user") {
                fields(block, {"type", "tool_use_id", "content", "is_error"}); flush();
                auto result = Json::array();
                if (boolean(block, "is_error")) result.push_back({{"type", "text"}, {"text", "Tool execution failed: "}});
                for (const auto& part : blocks(block.value("content", Json("")), true)) result.push_back(content(part));
                messages.push_back({{"role", "tool"}, {"tool_call_id", text(block.value("tool_use_id", Json()), "tool_result.tool_use_id")}, {"content", std::move(result)}});
            } else if (kind == "thinking" && role == "assistant") {
                fields(block, {"type", "thinking", "signature"});
                reasoning += text(block.value("thinking", Json()), "thinking", true);
                if (block.contains("signature")) text(block["signature"], "signature", true);
            } else parts.push_back(content(block));
        }
        flush();
    }
    converted["messages"] = std::move(messages);
    const auto tools = body.value("tools", Json::array());
    if (!tools.is_array() || tools.size() > 128) throw ProtocolError(400, "invalid_request_error", "tools must contain at most 128 definitions");
    auto native_tools = Json::array(); std::set<std::string> names;
    static const std::regex identifier(R"(^[A-Za-z_][A-Za-z0-9_.-]{0,127}$)");
    for (const auto& tool : tools) {
        fields(tool, {"type", "name", "description", "input_schema"});
        if (tool.contains("type") && tool["type"] != "custom") throw ProtocolError(400, "unsupported_parameter", "only client tools are supported");
        const auto name = text(tool.value("name", Json()), "tools.name");
        if (!std::regex_match(name, identifier) || !names.insert(name).second) throw ProtocolError(400, "invalid_request_error", "invalid or duplicate tool name");
        if (!tool.contains("input_schema") || !tool["input_schema"].is_object()) throw ProtocolError(400, "invalid_request_error", "input_schema must be an object");
        native_tools.push_back({{"type", "function"}, {"function", {{"name", name},
            {"description", text(tool.value("description", Json("")), "tools.description", true)}, {"parameters", tool["input_schema"]}}}});
    }
    converted["tools"] = std::move(native_tools);
    const auto choice = body.value("tool_choice", Json{{"type", "auto"}});
    fields(choice, {"type", "name", "disable_parallel_tool_use"});
    if (boolean(choice, "disable_parallel_tool_use")) throw ProtocolError(400, "unsupported_parameter", "disable_parallel_tool_use is not supported");
    const auto kind = text(choice.value("type", Json()), "tool_choice.type");
    if (kind == "tool") converted["tool_choice"] = {{"type", "function"}, {"function", {{"name", text(choice.value("name", Json()), "tool_choice.name")}}}};
    else {
        if (choice.contains("name") || (kind != "auto" && kind != "none" && kind != "any")) throw ProtocolError(400, "invalid_request_error", "invalid tool_choice");
        converted["tool_choice"] = kind == "any" ? "required" : kind;
    }
    return mfq::protocol::parse_input(converted, true, defaults);
}

class Message {
public:
    Message(const std::string& id, const std::string& model, bool streaming) : streaming_(streaming),
        resource_{{"id", "msg_" + id}, {"type", "message"}, {"role", "assistant"}, {"model", model},
            {"content", Json::array()}, {"stop_reason", nullptr}, {"stop_sequence", nullptr},
            {"usage", {{"input_tokens", 0}, {"output_tokens", 0}}}} {}
    const Json& resource() const { return resource_; }
    std::vector<Json> append(const std::vector<common_chat_msg_diff>& diffs) {
        std::vector<Json> events;
        for (const auto& diff : diffs) {
            const auto& call = diff.tool_call_delta;
            retained_ += call.id.size() + call.name.size() + call.arguments.size();
            if (!streaming_) retained_ += diff.content_delta.size() + diff.reasoning_content_delta.size();
            if (retained_ > 16 * 1024 * 1024) throw ProtocolError(413, "output_limit", "response content or tool arguments exceed 16 MiB");
            add_text("thinking", diff.reasoning_content_delta, events);
            add_text("text", diff.content_delta, events);
            if (diff.tool_call_index == std::string::npos) continue;
            if (diff.tool_call_index >= 128) throw ProtocolError(502, "invalid_tool_output", "native tool index exceeds 128 calls");
            close_text(events);
            auto& tool = tools_[diff.tool_call_index];
            if (tool.started && ((!call.id.empty() && call.id != tool.call.id) || (!call.name.empty() && call.name != tool.call.name)))
                throw ProtocolError(502, "invalid_tool_output", "native tool identity changed after arguments started");
            if (!call.id.empty()) tool.call.id = call.id;
            if (!call.name.empty()) tool.call.name = call.name;
            tool.call.arguments += call.arguments;
            if (!tool.started) {
                if (!tool.call.id.empty() && !tool.call.name.empty() && !tool.call.arguments.empty()) start_tool(tool, events);
            } else if (!call.arguments.empty()) emit(events, {{"type", "content_block_delta"}, {"index", tool.index},
                {"delta", {{"type", "input_json_delta"}, {"partial_json", call.arguments}}}});
        }
        return events;
    }
    std::vector<Json> finish(const mfq::protocol::CompletionStream& completion) {
        if (completion.result.cancelled) throw ProtocolError(409, "request_cancelled", "native generation was cancelled");
        std::vector<Json> events; close_text(events);
        for (auto& [id, tool] : tools_) {
            const auto arguments = Json::parse(tool.call.arguments.empty() ? "{}" : tool.call.arguments, nullptr, false);
            if (!arguments.is_object() || tool.call.id.empty() || tool.call.name.empty()) throw ProtocolError(502, "invalid_tool_output", "native tool call is incomplete or has invalid input");
            if (!tool.started) start_tool(tool, events);
            if (!streaming_) resource_["content"][tool.index]["input"] = arguments;
            emit(events, {{"type", "content_block_stop"}, {"index", tool.index}});
        }
        const auto& reason = completion.result.finish_reason;
        if (reason != "stop" && reason != "length" && reason != "tool_calls") throw ProtocolError(502, "invalid_terminal", "unsupported native finish reason");
        resource_["stop_reason"] = tools_.empty() ? (reason == "length" ? "max_tokens" : "end_turn") : "tool_use";
        resource_["usage"] = {{"input_tokens", completion.prompt_tokens}, {"output_tokens", completion.result.completion_tokens}};
        emit(events, {{"type", "message_delta"}, {"delta", {{"stop_reason", resource_["stop_reason"]}, {"stop_sequence", nullptr}}}, {"usage", resource_["usage"]}});
        emit(events, {{"type", "message_stop"}});
        return events;
    }
private:
    struct Tool { common_chat_tool_call call; std::size_t index = 0; bool started = false; };
    void emit(std::vector<Json>& events, Json value) { if (streaming_) events.push_back(std::move(value)); }
    void close_text(std::vector<Json>& events) {
        if (active_.empty()) return;
        if (active_ == "thinking") emit(events, {{"type", "content_block_delta"}, {"index", text_index_}, {"delta", {{"type", "signature_delta"}, {"signature", ""}}}});
        emit(events, {{"type", "content_block_stop"}, {"index", text_index_}}); active_.clear();
    }
    void add_text(const std::string& kind, const std::string& value, std::vector<Json>& events) {
        if (value.empty()) return;
        if (active_ != kind) {
            close_text(events); active_ = kind; text_index_ = count_++;
            Json block{{"type", kind}, {kind, ""}};
            emit(events, {{"type", "content_block_start"}, {"index", text_index_}, {"content_block", block}});
            if (!streaming_) {
                if (kind == "thinking") block["signature"] = "";
                resource_["content"].push_back(std::move(block));
            }
        }
        emit(events, {{"type", "content_block_delta"}, {"index", text_index_}, {"delta", {{"type", kind + "_delta"}, {kind, value}}}});
        if (!streaming_) resource_["content"][text_index_][kind].get_ref<std::string&>() += value;
    }
    void start_tool(Tool& tool, std::vector<Json>& events) {
        tool.index = count_++; tool.started = true;
        Json block{{"type", "tool_use"}, {"id", tool.call.id}, {"name", tool.call.name}, {"input", Json::object()}};
        emit(events, {{"type", "content_block_start"}, {"index", tool.index}, {"content_block", block}});
        if (!streaming_) resource_["content"].push_back(std::move(block));
        if (!tool.call.arguments.empty()) emit(events, {{"type", "content_block_delta"}, {"index", tool.index},
            {"delta", {{"type", "input_json_delta"}, {"partial_json", tool.call.arguments}}}});
    }
    bool streaming_;
    Json resource_;
    std::string active_;
    std::size_t count_ = 0, text_index_ = 0, retained_ = 0;
    std::map<std::size_t, Tool> tools_;
};
struct Generation {
    StreamSlot slot;
    std::unique_ptr<mfq::protocol::CompletionStream> completion;
    Message message;
    Generation(std::shared_ptr<std::atomic<std::size_t>> active, const std::string& id, const std::string& model, bool stream)
        : slot(std::move(active)), message(id, model, stream) {}
};
bool send(httplib::DataSink& sink, const Json& event) {
    const auto frame = "event: " + event.at("type").get<std::string>() + "\ndata: " + event.dump() + "\n\n";
    return sink.write(frame.data(), frame.size());
}
}

void register_anthropic(httplib::Server& http, ModelInstances& models, std::shared_ptr<std::atomic<std::size_t>> active) {
    http.Post("/v1/messages/count_tokens", [](const httplib::Request&, httplib::Response&) {
        throw ProtocolError(501, "unsupported_parameter", "native runtime does not expose exact prompt token counting");
    });
    http.Post("/v1/messages", [&, active](const httplib::Request& request, httplib::Response& response) {
        const auto version = request.get_header_value("Anthropic-Version");
        if (!version.empty() && version != "2023-06-01") throw ProtocolError(400, "unsupported_parameter", "supported Anthropic-Version is 2023-06-01");
        const auto body = parse_json(request);
        if (!body.is_object()) throw ProtocolError(400, "invalid_request_error", "request must be an object");
        const auto model = text(body.value("model", Json()), "model");
        MfqSamplingParams validation; validation.temperature = 0; validation.top_p = 1; validation.top_k = 1;
        parse_messages(body, validation);
        const bool streaming = boolean(body, "stream");
        const auto id = normalize_uuid({});
        auto generation = std::make_shared<Generation>(active, id, model, streaming);
        const auto instance = models.acquire(model, request.is_connection_closed);
        auto input = parse_messages(body, instance->sampling());
        generation->completion = std::make_unique<mfq::protocol::CompletionStream>(
            instance->submit({id, std::move(input)}), true, [instance, id] { instance->cancel(id); });
        response.set_header("X-Request-ID", id); response.set_header("request-id", id);
        const auto disconnected = request.is_connection_closed;
        if (!streaming) {
            while (auto diffs = generation->completion->next()) {
                if (disconnected()) { generation->completion->cancel(); return; }
                generation->message.append(*diffs);
            }
            generation->message.finish(*generation->completion);
            set_json(response, generation->message.resource()); return;
        }
        response.set_header("Cache-Control", "no-cache"); response.set_header("X-Accel-Buffering", "no");
        response.set_chunked_content_provider("text/event-stream; charset=utf-8",
            [generation, disconnected](std::size_t, httplib::DataSink& sink) {
                auto& completion = *generation->completion;
                try {
                    if (!send(sink, {{"type", "message_start"}, {"message", generation->message.resource()}})) { completion.cancel(); return false; }
                    auto heartbeat = engine::Clock::now();
                    while (auto diffs = completion.next()) {
                        if (disconnected()) { completion.cancel(); return false; }
                        for (const auto& event : generation->message.append(*diffs)) if (!send(sink, event)) { completion.cancel(); return false; }
                        if (engine::Clock::now() - heartbeat >= std::chrono::seconds(15)) {
                            if (!send(sink, {{"type", "ping"}})) { completion.cancel(); return false; }
                            heartbeat = engine::Clock::now();
                        }
                    }
                    for (const auto& event : generation->message.finish(completion)) if (!send(sink, event)) return false;
                } catch (...) {
                    send(sink, compat_error(describe_failure(std::current_exception()), true));
                }
                sink.done(); return false;
            }, [generation](bool) {});
    });
}
}
