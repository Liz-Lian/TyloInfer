#include "sessions.h"
#include "storage/sqlite.h"
#include "models/errors.h"
#include "instance.h"
#include "protocol.h"
#include "mfq_paged_prefix_cache.h"

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>

namespace mfq::server {
namespace {
using Json = nlohmann::json;
using Error = Sessions::Error;
using sqlite::Statement;
constexpr std::size_t history_limit = 16 * 1024 * 1024;
constexpr std::size_t output_limit = 16 * 1024 * 1024;

void fields(const Json& body, std::initializer_list<const char*> names) {
    if (!body.is_object()) throw Error(Error::invalid, "invalid_request", "request must be an object");
    for (const auto& field : body.items())
        if (std::find(names.begin(), names.end(), field.key()) == names.end())
            throw Error(Error::invalid, "invalid_request", "unknown field: " + field.key());
}
std::string text(const Json& value, std::size_t maximum, bool empty = true) {
    if (!value.is_string()) throw Error(Error::invalid, "invalid_request", "expected a string");
    auto result = value.get<std::string>();
    const auto length = std::count_if(result.begin(), result.end(), [](unsigned char c) { return (c & 192) != 128; });
    if ((!empty && result.empty()) || std::size_t(length) > maximum)
        throw Error(Error::invalid, "invalid_request", "string length is out of range");
    return result;
}
std::int64_t integer(const Json& value) {
    if (!value.is_number_integer() || (value.is_number_unsigned() && value.get<std::uint64_t>() > std::uint64_t(INT64_MAX)) ||
        value.get<std::int64_t>() < 0)
        throw Error(Error::invalid, "invalid_request", "expected a nonnegative integer");
    return value.get<std::int64_t>();
}
bool boolean(const Json& body, const char* key, bool fallback) {
    if (!body.contains(key)) return fallback;
    if (!body[key].is_boolean()) throw Error(Error::invalid, "invalid_request", std::string(key) + " must be boolean");
    return body[key].get<bool>();
}
std::string timestamp() {
    const auto value = std::time(nullptr);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &value);
#else
    gmtime_r(&value, &utc);
#endif
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S+00:00");
    return out.str();
}
Json parse_column(const Statement& statement, int column) {
    const auto value = statement.column(column);
    return value.is_null() ? value : Json::parse(value.get<std::string>());
}
Json session(sqlite3* db, const std::string& id) {
    Statement query(db, "SELECT id,model,mode,state,revision,title,runtime_instance_id,created_at,updated_at,metadata_json FROM sessions WHERE id=?");
    query.bind(1, id);
    if (!query.step()) throw Error(Error::not_found, "session_not_found", "session not found");
    return {{"id", query.column(0)}, {"model", query.column(1)}, {"mode", query.column(2)}, {"state", query.column(3)},
        {"revision", std::stoll(query.column(4).get<std::string>())}, {"title", query.column(5)},
        {"runtime_instance_id", query.column(6)}, {"created_at", query.column(7)}, {"updated_at", query.column(8)}, {"metadata", parse_column(query, 9)}};
}
Json history(sqlite3* db, const std::string& id) {
    Statement query(db, "SELECT m.id,m.role,m.parts_json,m.parent_id,m.created_at FROM session_messages sm "
        "JOIN messages m ON m.id=sm.message_id WHERE sm.session_id=? ORDER BY sm.ordinal");
    query.bind(1, id);
    auto result = Json::array();
    std::size_t bytes = 0;
    while (query.step()) {
        const auto parts = query.column(2).get<std::string>();
        bytes += parts.size();
        if (bytes > history_limit || result.size() >= 10000)
            throw Error(Error::too_large, "history_limit", "session history exceeds 16 MiB or 10000 messages");
        result.push_back({{"id", query.column(0)}, {"role", query.column(1)}, {"parts", Json::parse(parts)},
            {"parent_id", query.column(3)}, {"created_at", query.column(4)}});
    }
    return result;
}
Json response_row(Statement& query) {
    return {{"id", query.column(0)}, {"request_id", query.column(1)}, {"session_id", query.column(2)}, {"status", query.column(3)},
        {"output_message_id", query.column(4)}, {"output", parse_column(query, 5)}, {"finish_reason", query.column(6)},
        {"usage", parse_column(query, 7)}, {"performance", parse_column(query, 8)}, {"settings", parse_column(query, 9)},
        {"error", parse_column(query, 10)}, {"created_at", query.column(11)}, {"completed_at", query.column(12)}};
}
constexpr const char* response_columns = "SELECT id,request_id,session_id,status,output_message_id,output_json,finish_reason,usage_json,performance_json,settings_json,error_json,created_at,completed_at FROM responses ";
Json response(sqlite3* db, const std::string& id) {
    Statement query(db, (std::string(response_columns) + "WHERE id=?").c_str());
    query.bind(1, id);
    if (!query.step()) throw Error(Error::not_found, "response_not_found", "response not found");
    return response_row(query);
}
void editable(const Json& current, const Json& expected) {
    if (!expected.is_null() && integer(expected) != current.at("revision").get<std::int64_t>())
        throw Error(Error::conflict, "revision_conflict", "session revision changed",
            {{"expected_revision", expected}, {"actual_revision", current.at("revision")}});
    if (current.at("state") != "idle" && current.at("state") != "error" && current.at("state") != "interrupted")
        throw Error(Error::conflict, "response_in_progress", "session has an active response");
}
Json normalize_parts(Json parts) {
    if (!parts.is_array() || parts.empty() || parts.size() > 1024)
        throw Error(Error::invalid, "invalid_request", "parts must contain 1-1024 entries");
    for (auto& part : parts) {
        if (!part.is_object() || !part.contains("type")) throw Error(Error::invalid, "invalid_request", "part type is required");
        const auto type = text(part.at("type"), 32, false);
        if (type == "text" || type == "reasoning" || type == "transcript") {
            if (type == "transcript") {
                fields(part, {"type", "text", "language", "start_ms", "end_ms"});
                for (const auto* key : {"language", "start_ms", "end_ms"}) if (!part.contains(key)) part[key] = nullptr;
                if (!part["language"].is_null() && text(part["language"], 35).size() < 2)
                    throw Error(Error::invalid, "invalid_request", "transcript language must contain 2-35 characters");
                for (const auto* key : {"start_ms", "end_ms"}) if (!part[key].is_null()) integer(part[key]);
            } else fields(part, {"type", "text"});
            text(part.value("text", Json()), history_limit);
        } else if (type == "tool_call") {
            fields(part, {"type", "call_id", "name", "arguments"});
            text(part.value("call_id", Json()), 255, false); text(part.value("name", Json()), 255, false);
            if (!part.contains("arguments")) part["arguments"] = Json::object();
            if (!part["arguments"].is_object())
                throw Error(Error::invalid, "invalid_request", "tool arguments must be an object");
        } else if (type == "tool_result") {
            fields(part, {"type", "call_id", "result", "is_error"});
            text(part.value("call_id", Json()), 255, false);
            if (!part.contains("is_error")) part["is_error"] = false;
            if (!part.contains("result") || !part["is_error"].is_boolean())
                throw Error(Error::invalid, "invalid_request", "tool result and is_error are required");
        } else if (type == "image" || type == "video" || type == "audio" || type == "generated_audio" || type == "document") {
            throw Error(Error::unavailable, "media_unavailable", "native session media validation and archives are not implemented");
        } else throw Error(Error::invalid, "invalid_request", "unknown content part type");
    }
    if (parts.dump().size() > history_limit) throw Error(Error::too_large, "history_limit", "message exceeds 16 MiB");
    return parts;
}
void validate_role(const Json& role) {
    if (role != "system" && role != "user" && role != "assistant" && role != "tool")
        throw Error(Error::invalid, "invalid_request", "invalid message role");
}
void validate_mode(const Json& mode) {
    if (mode != "text") throw Error(Error::unavailable, "session_mode_unavailable", "native sessions currently support text mode only");
}
Json add_message(sqlite3* db, const std::string& id, const Json& role, const Json& parts, const std::string& when) {
    Statement previous(db, "SELECT message_id FROM session_messages WHERE session_id=? ORDER BY ordinal DESC LIMIT 1");
    previous.bind(1, id);
    const auto parent = previous.step() ? previous.column(0) : Json(nullptr);
    const auto message_id = normalize_uuid({});
    Statement insert(db, "INSERT INTO messages(id,role,parts_json,parent_id,created_at) VALUES(?,?,?,?,?)");
    insert.bind(1, message_id); insert.bind(2, role); insert.bind(3, parts.dump()); insert.bind(4, parent); insert.bind(5, when); insert.step();
    Statement link(db, "INSERT INTO session_messages(session_id,ordinal,message_id) SELECT ?,COALESCE(MAX(ordinal),0)+1,? FROM session_messages WHERE session_id=?");
    link.bind(1, id); link.bind(2, message_id); link.bind(3, id); link.step();
    Statement update(db, "UPDATE sessions SET revision=revision+1,updated_at=? WHERE id=?");
    update.bind(1, when); update.bind(2, id); update.step();
    return {{"id", message_id}, {"role", role}, {"parts", parts}, {"parent_id", parent}, {"created_at", when}};
}
Json create_session(sqlite3* db, const Json& body) {
    fields(body, {"model", "mode", "title", "metadata"});
    const auto model = text(body.value("model", Json("")), 255);
    const auto mode = body.value("mode", Json("text")); validate_mode(mode);
    const auto title = body.value("title", Json()); if (!title.is_null()) text(title, 512);
    const auto metadata = body.value("metadata", Json::object());
    if (!metadata.is_object()) throw Error(Error::invalid, "invalid_request", "metadata must be an object");
    const auto id = normalize_uuid({}), now = timestamp();
    Statement insert(db, "INSERT INTO sessions(id,model,mode,state,revision,title,created_at,updated_at,metadata_json) VALUES(?,?,?,'idle',0,?,?,?,?)");
    insert.bind(1, id); insert.bind(2, model); insert.bind(3, mode); insert.bind(4, title);
    insert.bind(5, now); insert.bind(6, now); insert.bind(7, metadata.dump()); insert.step();
    return session(db, id);
}
Json backend_message(const Json& message, bool reasoning) {
    const auto parts = normalize_parts(message.at("parts"));
    Json result{{"role", message.at("role")}, {"content", ""}};
    std::string content, thought;
    for (const auto& part : parts) {
        const auto& type = part.at("type");
        if (type == "text" || type == "transcript") content += part.at("text").get<std::string>();
        else if (type == "reasoning" && reasoning) thought += part.at("text").get<std::string>();
        else if (type == "tool_call") {
            if (!result.contains("tool_calls")) result["tool_calls"] = Json::array();
            result["tool_calls"].push_back({{"id", part.at("call_id")}, {"type", "function"},
                {"function", {{"name", part.at("name")}, {"arguments", part.at("arguments").dump()}}}});
        } else if (type == "tool_result") {
            result["tool_call_id"] = part.at("call_id");
            content += part.at("result").is_string() ? part.at("result").get<std::string>() : part.at("result").dump();
        }
    }
    result["content"] = std::move(content);
    if (!thought.empty()) result["reasoning_content"] = std::move(thought);
    return result;
}
Json settings(Json body) {
    fields(body, {"sampling", "model", "system_prompt", "include_reasoning_history", "input_role", "tools", "tool_choice", "response_format"});
    if (!body.contains("model")) body["model"] = nullptr;
    if (!body["model"].is_null()) text(body["model"], 255, false);
    if (!body.contains("system_prompt")) body["system_prompt"] = nullptr;
    if (!body["system_prompt"].is_null()) text(body["system_prompt"], 32768);
    body["include_reasoning_history"] = boolean(body, "include_reasoning_history", true);
    if (!body.contains("input_role")) body["input_role"] = "user";
    if (body["input_role"] != "user" && body["input_role"] != "tool") throw Error(Error::invalid, "invalid_request", "input_role must be user or tool");
    Json sampling = body.value("sampling", Json::object());
    fields(sampling, {"max_tokens", "temperature", "top_k", "top_p", "presence_penalty", "frequency_penalty", "repetition_penalty", "seed",
        "enable_thinking", "enable_vision", "enable_mtp", "mtp_max_draft_tokens", "reasoning_effort"});
    for (const auto* key : {"max_tokens", "top_k", "mtp_max_draft_tokens"})
        if (sampling.contains(key) && integer(sampling[key]) > INT32_MAX)
            throw Error(Error::invalid, "invalid_request", "sampling integer is out of range");
    for (const auto* key : {"temperature", "top_p", "presence_penalty", "frequency_penalty", "repetition_penalty"}) {
        if (!sampling.contains(key)) continue;
        if (!sampling[key].is_number()) throw Error(Error::invalid, "invalid_request", "sampling value must be numeric");
        sampling[key] = sampling[key].get<double>();
    }
    for (const auto* key : {"enable_thinking", "enable_vision", "enable_mtp"}) boolean(sampling, key, true);
    if (sampling.contains("seed") && !sampling["seed"].is_null()) integer(sampling["seed"]);
    if (sampling.contains("reasoning_effort") && !sampling["reasoning_effort"].is_null())
        throw Error(Error::unavailable, "unsupported_parameter", "reasoning_effort has no native request control yet");
    body["sampling"] = std::move(sampling);
    if (!body.contains("tools")) body["tools"] = Json::array();
    if (!body["tools"].is_array() || body["tools"].size() > 128) throw Error(Error::invalid, "invalid_request", "tools must contain at most 128 entries");
    static const std::regex identifier(R"(^[A-Za-z_][A-Za-z0-9_.-]{0,127}$)");
    std::vector<std::string> names;
    for (auto& tool : body["tools"]) {
        fields(tool, {"type", "function"});
        if (!tool.contains("type")) tool["type"] = "function";
        if (tool["type"] != "function" || !tool.contains("function"))
            throw Error(Error::invalid, "invalid_request", "invalid tool definition");
        auto& function = tool["function"];
        fields(function, {"name", "description", "parameters", "strict"});
        const auto name = text(function.value("name", Json()), 128, false);
        if (!std::regex_match(name, identifier) || std::find(names.begin(), names.end(), name) != names.end())
            throw Error(Error::invalid, "invalid_request", "invalid or duplicate tool name");
        names.push_back(name);
        if (!function.contains("description")) function["description"] = nullptr;
        if (!function["description"].is_null()) text(function["description"], 8192);
        if (!function.contains("parameters")) function["parameters"] = {{"type", "object"}};
        if (!function["parameters"].is_object()) throw Error(Error::invalid, "invalid_request", "tool parameters must be an object");
        if (!function.contains("strict")) function["strict"] = false;
        if (!function["strict"].is_null()) boolean(function, "strict", false);
    }
    if (!body.contains("tool_choice")) body["tool_choice"] = "auto";
    auto& choice = body["tool_choice"];
    if (choice.is_object()) {
        fields(choice, {"type", "function"});
        if (!choice.contains("type")) choice["type"] = "function";
        if (choice["type"] != "function" || !choice.contains("function")) throw Error(Error::invalid, "invalid_request", "invalid tool_choice");
        fields(choice["function"], {"name"}); text(choice["function"].value("name", Json()), 128, false);
    } else if (choice != "auto" && choice != "none" && choice != "required") throw Error(Error::invalid, "invalid_request", "invalid tool_choice");
    if (!body.contains("response_format")) body["response_format"] = {{"type", "text"}};
    auto& format = body["response_format"];
    if (!format.is_object() || !format.contains("type")) throw Error(Error::invalid, "invalid_request", "response_format requires type");
    if (format["type"] == "text" || format["type"] == "json_object") fields(format, {"type"});
    else if (format["type"] == "json_schema") {
        fields(format, {"type", "json_schema"});
        if (!format.contains("json_schema")) throw Error(Error::invalid, "invalid_request", "json_schema is required");
        auto& schema = format["json_schema"];
        fields(schema, {"name", "description", "schema", "strict"});
        if (!std::regex_match(text(schema.value("name", Json()), 128, false), identifier) || !schema.contains("schema") || !schema["schema"].is_object())
            throw Error(Error::invalid, "invalid_request", "invalid named JSON schema");
        if (!schema.contains("description")) schema["description"] = nullptr;
        if (!schema["description"].is_null()) text(schema["description"], 8192);
        schema["strict"] = boolean(schema, "strict", true);
    } else throw Error(Error::invalid, "invalid_request", "unsupported response_format");
    return body;
}
engine::InferenceInput native_input(const Json& options, Json messages, const std::string& id, const MfqSamplingParams& defaults) {
    auto body = options.at("sampling");
    body["messages"] = std::move(messages);
    body["tools"] = options.at("tools"); body["tool_choice"] = options.at("tool_choice");
    for (auto& tool : body["tools"]) if (tool["function"]["description"].is_null()) tool["function"].erase("description");
    body["response_format"] = options.at("response_format"); body["stream"] = true;
    body["mfq_session_id"] = id;
    try { return mfq::protocol::parse_input(body, true, defaults); }
    catch (const mfq::protocol::ApiError& error) { throw Error(Error::invalid, error.type, error.what()); }
}
Json preset(sqlite3* db, const std::string& id) {
    Statement query(db, "SELECT id,name,model,mode,settings_json,context_size,metadata_json,created_at,updated_at FROM generation_presets WHERE id=?");
    query.bind(1, id);
    if (!query.step()) throw Error(Error::not_found, "generation_preset_not_found", "generation preset not found");
    return {{"id", query.column(0)}, {"name", query.column(1)}, {"model", query.column(2)}, {"mode", query.column(3)},
        {"settings", parse_column(query, 4)}, {"context_size", query.column(5).is_null() ? Json() : Json(std::stoll(query.column(5).get<std::string>()))},
        {"metadata", parse_column(query, 6)}, {"created_at", query.column(7)}, {"updated_at", query.column(8)}};
}
void append_record(Json& data, Json record, std::size_t& bytes) {
    bytes += record.dump().size();
    if (bytes > history_limit) throw Error(Error::too_large, "result_limit", "result page exceeds 16 MiB; request fewer records");
    data.push_back(std::move(record));
}
Json detail(std::string code, std::string message, bool retryable = false) {
    return {{"code", std::move(code)}, {"message", std::move(message)}, {"retryable", retryable}, {"details", Json::object()}};
}
} // namespace

struct Sessions::Reply::State {
    Sessions* owner;
    mutable std::mutex mutex;
    std::mutex reader;
    std::atomic<bool> cancelled{false};
    std::shared_ptr<ModelInstance> instance;
    std::shared_ptr<MfqScheduledRequest> outbox;
    Json resource, error, messages;
    common_chat_msg output;
    engine::InferenceMetrics metrics;
    engine::UsageUpdate usage;
    MfqSamplingParams sampling;
    std::string execution, status, native_id;
    bool terminal = false;
    std::exception_ptr save_failure;
    std::size_t bytes = 0;
};

Sessions::Sessions(Storage& storage, ModelInstances& models) : storage_(storage), models_(models) {
    auto access = storage_.access();
    sqlite::Transaction transaction(access.database);
    Statement recover(access.database, "UPDATE sessions SET state='interrupted',runtime_instance_id=NULL,updated_at=? "
        "WHERE state IN ('processing','listening','speaking','reconnecting') OR id IN (SELECT session_id FROM responses WHERE status='running')");
    recover.bind(1, timestamp()); recover.step();
    Statement pending(access.database, "UPDATE responses SET status='failed',error_json=?,completed_at=? WHERE status='running'");
    pending.bind(1, detail("response_interrupted", "server restarted before reply persistence completed").dump());
    pending.bind(2, timestamp()); pending.step();
    transaction.commit();
}
Sessions::~Sessions() { try { shutdown(); } catch (const std::exception& error) { std::cerr << "session shutdown: " << error.what() << '\n'; } }
Sessions::Json Sessions::create(const Json& request) {
    auto access = storage_.access(); sqlite::Transaction transaction(access.database);
    auto result = create_session(access.database, request); transaction.commit(); return result;
}
Sessions::Json Sessions::get(const std::string& id) const { auto access = storage_.access(); return session(access.database, id); }
Sessions::Json Sessions::list(std::size_t limit, std::size_t offset) const {
    if (!limit || limit > 200 || offset > std::size_t(INT64_MAX)) throw Error(Error::invalid, "invalid_request", "invalid pagination");
    auto access = storage_.access();
    Statement query(access.database, "SELECT id FROM sessions ORDER BY updated_at DESC,id DESC LIMIT ? OFFSET ?");
    query.bind(1, std::to_string(limit)); query.bind(2, std::to_string(offset));
    auto data = Json::array(); std::size_t bytes = 0;
    while (query.step()) append_record(data, session(access.database, query.column(0).get<std::string>()), bytes);
    return {{"data", std::move(data)}};
}
Sessions::Json Sessions::update(const std::string& id, const Json& body) {
    fields(body, {"title", "mode", "metadata"});
    if (body.empty()) throw Error(Error::invalid, "invalid_request", "at least one field is required");
    auto access = storage_.access(); sqlite::Transaction transaction(access.database);
    auto current = session(access.database, id);
    if (body.contains("title")) { if (!body["title"].is_null()) text(body["title"], 512); current["title"] = body["title"]; }
    if (body.contains("mode")) { editable(current, nullptr); validate_mode(body["mode"]); current["mode"] = body["mode"]; }
    if (body.contains("metadata")) {
        if (!body["metadata"].is_object()) throw Error(Error::invalid, "invalid_request", "metadata must be an object");
        current["metadata"] = body["metadata"];
    }
    Statement query(access.database, "UPDATE sessions SET title=?,mode=?,metadata_json=?,updated_at=? WHERE id=?");
    query.bind(1, current["title"]); query.bind(2, current["mode"]); query.bind(3, current["metadata"].dump());
    query.bind(4, timestamp()); query.bind(5, id); query.step();
    auto result = session(access.database, id); transaction.commit(); return result;
}
Sessions::Json Sessions::messages(const std::string& id) const {
    auto access = storage_.access(); session(access.database, id); return {{"data", history(access.database, id)}};
}
Sessions::Json Sessions::append(const std::string& id, const Json& body) {
    fields(body, {"expected_revision", "role", "parts"});
    const auto expected = integer(body.value("expected_revision", Json()));
    const auto role = body.value("role", Json()); validate_role(role);
    const auto parts = normalize_parts(body.value("parts", Json()));
    auto access = storage_.access(); sqlite::Transaction transaction(access.database);
    editable(session(access.database, id), expected);
    const auto message = add_message(access.database, id, role, parts, timestamp());
    history(access.database, id);
    auto result = Json{{"session", session(access.database, id)}, {"message", message}};
    transaction.commit(); return result;
}
Sessions::Json Sessions::fork(const std::string& id, const Json& body) {
    fields(body, {"at_message_id", "include_message", "title", "model"});
    const bool include = boolean(body, "include_message", true);
    auto access = storage_.access(); sqlite::Transaction transaction(access.database);
    const auto source = session(access.database, id);
    auto count = source.at("revision").get<std::int64_t>();
    if (body.contains("at_message_id") && !body["at_message_id"].is_null()) {
        Statement target(access.database, "SELECT ordinal FROM session_messages WHERE session_id=? AND message_id=?");
        target.bind(1, id); target.bind(2, normalize_uuid(text(body["at_message_id"], 36, false)));
        if (!target.step()) throw Error(Error::not_found, "message_not_found", "message is not in this session");
        count = std::stoll(target.column(0).get<std::string>()) - !include;
    } else if (!include) throw Error(Error::invalid, "invalid_request", "include_message=false requires at_message_id");
    Json options{{"model", source.at("model")}, {"mode", source.at("mode")}, {"title", source.at("title")}, {"metadata", source.at("metadata")}};
    if (body.contains("model") && !body["model"].is_null()) { text(body["model"], 255, false); options["model"] = body["model"]; }
    if (body.contains("title") && !body["title"].is_null()) options["title"] = body["title"];
    auto created = create_session(access.database, options);
    Statement copy(access.database, "INSERT INTO session_messages(session_id,ordinal,message_id) SELECT ?,ordinal,message_id FROM session_messages WHERE session_id=? AND ordinal<=?");
    copy.bind(1, created["id"]); copy.bind(2, id); copy.bind(3, std::to_string(count)); copy.step();
    Statement revision(access.database, "UPDATE sessions SET revision=? WHERE id=?");
    revision.bind(1, std::to_string(count)); revision.bind(2, created["id"]); revision.step();
    auto result = session(access.database, created["id"]); transaction.commit(); return result;
}
Sessions::Json Sessions::rewind(const std::string& id, const Json& body) {
    fields(body, {"expected_revision", "at_message_id", "include_message"});
    const auto expected = integer(body.value("expected_revision", Json()));
    const bool include = boolean(body, "include_message", true);
    auto access = storage_.access(); sqlite::Transaction transaction(access.database);
    editable(session(access.database, id), expected);
    Statement target(access.database, "SELECT ordinal FROM session_messages WHERE session_id=? AND message_id=?");
    target.bind(1, id); target.bind(2, normalize_uuid(text(body.value("at_message_id", Json()), 36, false)));
    if (!target.step()) throw Error(Error::not_found, "message_not_found", "message is not in this session");
    const auto count = std::stoll(target.column(0).get<std::string>()) - !include;
    Statement trim_responses(access.database, "DELETE FROM responses WHERE session_id=? AND (input_message_id IN "
        "(SELECT message_id FROM session_messages WHERE session_id=? AND ordinal>?) OR output_message_id IN "
        "(SELECT message_id FROM session_messages WHERE session_id=? AND ordinal>?))");
    trim_responses.bind(1, id); trim_responses.bind(2, id); trim_responses.bind(3, std::to_string(count));
    trim_responses.bind(4, id); trim_responses.bind(5, std::to_string(count)); trim_responses.step();
    Statement trim(access.database, "DELETE FROM session_messages WHERE session_id=? AND ordinal>?");
    trim.bind(1, id); trim.bind(2, std::to_string(count)); trim.step();
    Statement update(access.database, "UPDATE sessions SET revision=?,state='idle',updated_at=? WHERE id=?");
    update.bind(1, std::to_string(count)); update.bind(2, timestamp()); update.bind(3, id); update.step();
    auto result = session(access.database, id); transaction.commit(); return result;
}
void Sessions::erase(const std::string& id) {
    auto access = storage_.access(); sqlite::Transaction transaction(access.database);
    editable(session(access.database, id), nullptr);
    Statement query(access.database, "DELETE FROM sessions WHERE id=?"); query.bind(1, id); query.step(); transaction.commit();
}
Sessions::Json Sessions::export_archive(const std::string& id) const {
    auto access = storage_.access();
    const auto current = session(access.database, id);
    auto data = history(access.database, id);
    for (auto& message : data) { message["parts"] = normalize_parts(message.at("parts")); message.erase("id"); message.erase("parent_id"); }
    return {{"format", "mfq-session-v1"}, {"session", current}, {"messages", std::move(data)}, {"media", Json::array()}};
}
Sessions::Json Sessions::import_archive(const Json& archive) {
    fields(archive, {"format", "session", "messages", "media"});
    if (archive.value("format", Json()) != "mfq-session-v1" || !archive.contains("session") || !archive["session"].is_object() ||
        !archive.contains("messages") || !archive["messages"].is_array() || archive["messages"].size() > 10000)
        throw Error(Error::invalid, "invalid_request", "invalid session archive");
    if (archive.contains("media") && (!archive["media"].is_array() || !archive["media"].empty()))
        throw Error(Error::unavailable, "media_unavailable", "media archive import is not implemented");
    const auto& source = archive["session"];
    auto metadata = source.value("metadata", Json::object());
    if (!metadata.is_object()) throw Error(Error::invalid, "invalid_request", "metadata must be an object");
    metadata["imported_from"] = normalize_uuid(text(source.value("id", Json()), 36, false));
    static const std::regex date(R"(^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(\.\d{1,6})?(Z|[+-](\d{2}):(\d{2}))$)");
    auto access = storage_.access(); sqlite::Transaction transaction(access.database);
    auto created = create_session(access.database, {{"model", source.value("model", Json(""))}, {"mode", source.value("mode", Json("text"))},
        {"title", source.value("title", Json())}, {"metadata", std::move(metadata)}});
    const auto id = created.at("id").get<std::string>();
    for (const auto& message : archive["messages"]) {
        fields(message, {"role", "parts", "created_at"});
        const auto role = message.value("role", Json()); validate_role(role);
        const auto parts = normalize_parts(message.value("parts", Json()));
        const auto when = text(message.value("created_at", Json()), 32, false);
        std::smatch parsed;
        if (!std::regex_match(when, parsed, date)) throw Error(Error::invalid, "invalid_request", "created_at must be a timestamp with timezone");
        const int year = std::stoi(parsed[1]), month = std::stoi(parsed[2]), day = std::stoi(parsed[3]);
        constexpr int days[] = {31,28,31,30,31,30,31,31,30,31,30,31};
        const bool leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
        if (year < 1 || month < 1 || month > 12 || day < 1 || day > days[month - 1] + (month == 2 && leap) ||
            std::stoi(parsed[4]) > 23 || std::stoi(parsed[5]) > 59 || std::stoi(parsed[6]) > 59 ||
            (parsed[8] != "Z" && (std::stoi(parsed[9]) > 23 || std::stoi(parsed[10]) > 59)))
            throw Error(Error::invalid, "invalid_request", "created_at is not a valid calendar timestamp");
        add_message(access.database, id, role, parts, when);
    }
    history(access.database, id);
    auto result = Json{{"session", session(access.database, id)}, {"messages_imported", archive["messages"].size()}, {"media_imported", 0}};
    transaction.commit(); return result;
}
Sessions::Json Sessions::presets() const {
    auto access = storage_.access();
    Statement query(access.database, "SELECT id FROM generation_presets ORDER BY name COLLATE NOCASE,id");
    auto data = Json::array(); std::size_t bytes = 0;
    while (query.step()) append_record(data, preset(access.database, query.column(0).get<std::string>()), bytes);
    return {{"data", std::move(data)}};
}
Sessions::Json Sessions::put_preset(const Json& body, const std::string& existing) {
    fields(body, {"name", "model", "mode", "settings", "context_size", "metadata"});
    const auto name = text(body.value("name", Json()), 64, false);
    const auto model = body.value("model", Json()); if (!model.is_null()) text(model, 255, false);
    const auto mode = body.value("mode", Json()); if (!mode.is_null()) validate_mode(mode);
    const auto context = body.value("context_size", Json());
    if (!context.is_null() && (integer(context) < 512 || integer(context) > INT32_MAX)) throw Error(Error::invalid, "invalid_request", "invalid context_size");
    const auto metadata = body.value("metadata", Json::object());
    if (!metadata.is_object()) throw Error(Error::invalid, "invalid_request", "metadata must be an object");
    const auto options = settings(body.value("settings", Json()));
    // Missing values are model-dependent; validate only conflicts forced by explicit settings.
    MfqSamplingParams validation;
    validation.temperature = 0; validation.top_p = 1;
    native_input(options, Json::array({{{"role", "user"}, {"content", "validate"}}}), normalize_uuid({}), validation);
    auto access = storage_.access(); sqlite::Transaction transaction(access.database);
    if (!existing.empty()) preset(access.database, existing);
    const auto id = existing.empty() ? normalize_uuid({}) : existing;
    Statement duplicate(access.database, "SELECT id FROM generation_presets WHERE name=? AND id<>?");
    duplicate.bind(1, name); duplicate.bind(2, id);
    if (duplicate.step()) throw Error(Error::conflict, "generation_preset_conflict", "preset name already exists");
    const auto now = timestamp();
    Statement write(access.database, "INSERT INTO generation_presets(id,name,model,mode,settings_json,context_size,metadata_json,created_at,updated_at) "
        "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,model=excluded.model,mode=excluded.mode,settings_json=excluded.settings_json,"
        "context_size=excluded.context_size,metadata_json=excluded.metadata_json,updated_at=excluded.updated_at");
    write.bind(1, id); write.bind(2, name); write.bind(3, model); write.bind(4, mode); write.bind(5, options.dump());
    write.bind(6, context.is_null() ? context : Json(std::to_string(context.get<std::int64_t>())));
    write.bind(7, metadata.dump()); write.bind(8, now); write.bind(9, now); write.step();
    auto result = preset(access.database, id); transaction.commit(); return result;
}
void Sessions::erase_preset(const std::string& id) {
    auto access = storage_.access(); sqlite::Transaction transaction(access.database); preset(access.database, id);
    Statement query(access.database, "DELETE FROM generation_presets WHERE id=?"); query.bind(1, id); query.step(); transaction.commit();
}
Sessions::Json Sessions::responses(const std::string& id, std::size_t limit) const {
    if (!limit || limit > 1000) throw Error(Error::invalid, "invalid_request", "limit must be between 1 and 1000");
    auto access = storage_.access(); session(access.database, id);
    Statement query(access.database, (std::string(response_columns) + "WHERE session_id=? ORDER BY created_at,id LIMIT ?").c_str());
    query.bind(1, id); query.bind(2, std::to_string(limit));
    auto data = Json::array(); std::size_t bytes = 0;
    while (query.step()) append_record(data, response_row(query), bytes);
    return {{"data", std::move(data)}};
}

std::shared_ptr<Sessions::Reply> Sessions::start(const std::string& id, Json request) {
    fields(request, {"request_id", "model", "expected_revision", "input", "input_role", "sampling", "system_prompt", "include_reasoning_history", "tools", "tool_choice", "response_format", "stream"});
    const auto request_id = normalize_uuid(text(request.value("request_id", Json()), 36, false));
    const auto expected = integer(request.value("expected_revision", Json()));
    request["input"] = normalize_parts(request.value("input", Json()));
    boolean(request, "stream", true);
    auto config = request; for (const auto* key : {"request_id", "expected_revision", "input", "stream"}) config.erase(key);
    const auto options = settings(std::move(config));
    if (options["input_role"] == "tool") {
        if (request["input"].size() != 1 || request["input"][0]["type"] != "tool_result")
            throw Error(Error::invalid, "invalid_request", "tool input requires exactly one tool_result");
    } else for (const auto& part : request["input"])
        if (part["type"] == "tool_result") throw Error(Error::invalid, "invalid_request", "tool_result requires input_role=tool");
    Json canonical = options;
    if (canonical["model"].is_null()) canonical.erase("model");
    canonical["request_id"] = request_id; canonical["expected_revision"] = expected; canonical["input"] = request["input"];
    const auto fingerprint = mfq::cache::block_hash_hex(mfq::cache::sha256(canonical.dump(-1, ' ', true)));
    std::unique_lock guard(mutex_);
    if (stopping_) throw Error(Error::stopping, "server_stopping", "chat admission has stopped");
    auto access = storage_.access();
    sqlite::Transaction transaction(access.database);
    auto current = session(access.database, id);
    Statement existing(access.database, "SELECT id,request_fingerprint,status FROM responses WHERE session_id=? AND request_id=?");
    existing.bind(1, id); existing.bind(2, request_id);
    if (existing.step()) {
        if (existing.column(1) != fingerprint) throw Error(Error::conflict, "idempotency_conflict", "request_id was reused with different content");
        if (existing.column(2) == "running") throw Error(Error::conflict, "response_in_progress", "request is already running");
        auto state = std::make_unique<Reply::State>(); state->owner = this; state->terminal = true;
        state->resource = response(access.database, existing.column(0));
        transaction.commit();
        return std::shared_ptr<Reply>(new Reply(std::move(state)));
    }
    editable(current, expected); validate_mode(current.at("mode"));
    const auto model = options["model"].is_null() ? current.at("model").get<std::string>() : options["model"].get<std::string>();
    if (model.empty()) throw Error(Error::invalid, "model_required", "select a model before sending a message");
    auto data = history(access.database, id);
    auto messages = Json::array();
    if (!options["system_prompt"].is_null() && !options["system_prompt"].get<std::string>().empty())
        messages.push_back({{"role", "system"}, {"content", options["system_prompt"]}});
    for (const auto& message : data) messages.push_back(backend_message(message, options["include_reasoning_history"]));
    messages.push_back(backend_message({{"role", options["input_role"]}, {"parts", request["input"]}}, true));
    MfqSamplingParams validation; validation.temperature = 0; validation.top_p = 1; validation.top_k = 1;
    native_input(options, messages, id, validation);
    const auto response_id = normalize_uuid({}), now = timestamp();
    auto state = std::make_unique<Reply::State>(); state->owner = this; state->messages = std::move(messages); state->native_id = response_id;
    auto saved_settings = options; saved_settings["model"] = model;
    const auto message = add_message(access.database, id, options["input_role"], request["input"], now);
    history(access.database, id);
    Statement insert(access.database, "INSERT INTO responses(id,session_id,request_id,request_fingerprint,status,input_message_id,output_json,settings_json,created_at) VALUES(?,?,?,?,'running',?,'[]',?,?)");
    insert.bind(1, response_id); insert.bind(2, id); insert.bind(3, request_id); insert.bind(4, fingerprint);
    insert.bind(5, message["id"]); insert.bind(6, saved_settings.dump()); insert.bind(7, now); insert.step();
    Statement processing(access.database, "UPDATE sessions SET state='processing',model=?,runtime_instance_id=NULL WHERE id=?");
    processing.bind(1, model); processing.bind(2, id); processing.step();
    state->resource = response(access.database, response_id);
    auto reply = std::shared_ptr<Reply>(new Reply(std::move(state)));
    active_.insert_or_assign(id, reply);
    try { transaction.commit(); } catch (...) { active_.erase(id); reply->state_->terminal = true; throw; }
    return reply;
}
Sessions::Reply::Reply(std::unique_ptr<State> state) : state_(std::move(state)) {}
Sessions::Reply::~Reply() {
    if (!state_->terminal) try { cancel(); finish(); }
    catch (const std::exception& error) { std::cerr << "reply finalization: " << error.what() << '\n'; }
}
bool Sessions::Reply::cancel() {
    state_->cancelled = true;
    std::lock_guard guard(state_->mutex);
    if (state_->terminal) return false;
    return state_->outbox ? state_->instance->cancel(state_->native_id) : true;
}
Sessions::Json Sessions::Reply::snapshot() const {
    std::lock_guard guard(state_->mutex);
    if (state_->save_failure) std::rethrow_exception(state_->save_failure);
    return state_->resource;
}
Sessions::Json Sessions::Reply::finish() { while (next()) {} return snapshot(); }
std::optional<std::vector<common_chat_msg_diff>> Sessions::Reply::next(const std::function<bool()>& cancelled) {
    std::lock_guard reader(state_->reader);
    std::unique_lock guard(state_->mutex);
    if (state_->save_failure) std::rethrow_exception(state_->save_failure);
    if (state_->terminal) return {};
    const auto stopped = [&] { return state_->cancelled.load() || (cancelled && cancelled()); };
    if (!state_->outbox) {
        const auto fail = [&](std::string code, std::string message, bool interrupted = false) {
            if (!guard.owns_lock()) guard.lock();
            state_->execution = state_->status = state_->cancelled || interrupted ? "cancelled" : "failed";
            state_->error = state_->status == "cancelled" ? detail("client_cancelled", "reply cancelled before native submission") : detail(std::move(code), std::move(message));
        };
        try {
            guard.unlock();
            auto instance = state_->owner->models_.acquire(state_->resource.at("settings").at("model"), stopped);
            auto input = native_input(state_->resource.at("settings"), std::move(state_->messages), state_->resource.at("session_id"), instance->sampling());
            guard.lock();
            if (stopped()) fail("client_cancelled", "reply cancelled", true);
            else {
                {
                    auto access = state_->owner->storage_.access();
                    Statement update(access.database, "UPDATE sessions SET runtime_instance_id=? WHERE id=?");
                    update.bind(1, instance->id); update.bind(2, state_->resource["session_id"]); update.step();
                }
                state_->sampling = input.sampling; state_->instance = std::move(instance);
                state_->outbox = state_->instance->submit({state_->native_id, std::move(input)});
            }
        } catch (const ModelError& error) { fail(error.code, error.what(), error.kind == ModelError::Kind::cancelled); }
        catch (const Error& error) { fail(error.code, error.what()); }
        catch (const std::exception& error) { fail("request_start_failed", error.what()); }
        state_->messages = nullptr;
    }
    auto diffs = std::vector<common_chat_msg_diff>{};
    std::vector<engine::EngineEvent> events;
    if (state_->outbox) {
        if (stopped()) state_->instance->cancel(state_->native_id);
        guard.unlock();
        events = state_->outbox->wait_for(std::chrono::milliseconds(100));
        guard.lock();
    }
    for (auto& event : events) {
        std::visit([&](auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, engine::OutputDelta>) {
                for (auto& diff : value.diffs) {
                    state_->bytes += diff.content_delta.size() + diff.reasoning_content_delta.size() + diff.tool_call_delta.id.size() + diff.tool_call_delta.name.size() + diff.tool_call_delta.arguments.size();
                    if (state_->bytes > output_limit || (diff.tool_call_index != std::string::npos && diff.tool_call_index >= 128)) {
                        if (state_->error.is_null()) {
                            state_->error = detail("output_limit", "persisted reply exceeds 16 MiB or 128 tool calls");
                            state_->instance->cancel(state_->native_id);
                        }
                        continue;
                    }
                    if (!state_->error.is_null()) continue;
                    state_->output.content += diff.content_delta; state_->output.reasoning_content += diff.reasoning_content_delta;
                    if (diff.tool_call_index != std::string::npos) {
                        if (state_->output.tool_calls.size() <= diff.tool_call_index) state_->output.tool_calls.resize(diff.tool_call_index + 1);
                        auto& call = state_->output.tool_calls[diff.tool_call_index];
                        // 引擎的名称和 ID 是完整值，只有 arguments 是增量。
                        if (!diff.tool_call_delta.id.empty()) call.id = diff.tool_call_delta.id;
                        if (!diff.tool_call_delta.name.empty()) call.name = diff.tool_call_delta.name;
                        call.arguments += diff.tool_call_delta.arguments;
                    }
                    diffs.push_back(std::move(diff));
                }
            } else if constexpr (std::is_same_v<T, engine::Completed> || std::is_same_v<T, engine::Cancelled>) {
                state_->usage = value.usage; state_->metrics = value.metrics;
                if constexpr (std::is_same_v<T, engine::Completed>) { state_->execution = state_->status = "completed"; state_->resource["finish_reason"] = value.finish_reason; }
                else { state_->execution = state_->status = "cancelled"; state_->error = state_->error.is_null() ? detail("client_cancelled", "native request cancelled") : state_->error; }
                if (state_->bytes > output_limit || (!state_->error.is_null() && state_->error["code"] == "output_limit")) state_->status = "failed";
            } else if constexpr (std::is_same_v<T, engine::Failed>) {
                state_->execution = state_->status = "failed"; state_->error = detail(value.code, value.message, value.retryable);
            }
        }, event.data);
    }
    if (!state_->execution.empty()) {
        state_->terminal = true;
        try { state_->owner->save(*state_); }
        catch (...) {
            state_->save_failure = std::current_exception();
            state_->owner->release(state_->resource.at("session_id"), state_.get()); throw;
        }
        state_->owner->release(state_->resource.at("session_id"), state_.get());
    }
    return diffs;
}
void Sessions::save(Reply::State& state) {
    Json output = Json::array();
    if (!state.output.reasoning_content.empty()) output.push_back({{"type", "reasoning"}, {"text", state.output.reasoning_content}});
    if (!state.output.content.empty()) output.push_back({{"type", "text"}, {"text", state.output.content}});
    for (const auto& call : state.output.tool_calls) {
        auto arguments = Json::parse(call.arguments, nullptr, false);
        if (!arguments.is_object() || call.id.empty() || call.name.empty()) {
            if (state.status == "completed") {
                state.status = "failed"; state.error = detail("invalid_tool_output", "native tool call is incomplete or has invalid arguments");
            }
            break;
        }
        output.push_back({{"type", "tool_call"}, {"call_id", call.id}, {"name", call.name}, {"arguments", std::move(arguments)}});
    }
    if (output.empty() && state.status == "completed") output.push_back({{"type", "text"}, {"text", ""}});
    if (output.dump().size() > output_limit) {
        state.status = "failed"; state.error = detail("output_limit", "encoded reply exceeds 16 MiB"); output = Json::array();
    }
    const auto id = state.resource.at("session_id").get<std::string>();
    try {
        auto access = storage_.access(); sqlite::Transaction transaction(access.database);
        Json message_id;
        if (state.status == "completed") {
            const auto existing = history(access.database, id);
            std::size_t bytes = output.dump().size();
            for (const auto& message : existing) bytes += message.at("parts").dump().size();
            if (bytes > history_limit || existing.size() >= 10000) {
                state.status = "failed"; state.error = detail("history_limit", "reply cannot fit in session history; fork or rewind first");
                state.error["details"]["execution_status"] = "completed";
            } else {
                const auto message = add_message(access.database, id, "assistant", output, timestamp()); message_id = message.at("id");
            }
        }
        Json usage, performance;
        if (state.outbox) {
            engine::InferenceResult result; result.completion_tokens = static_cast<int32_t>(state.usage.completion_tokens);
            usage = mfq::protocol::usage_json(state.usage.prompt_tokens, result.completion_tokens);
            auto metrics = mfq::protocol::request_metric_values_json(mfq::protocol::request_metric_values(result, state.metrics), state.sampling);
            metrics["sampling"]["enable_vision"] = state.sampling.enable_vision;
            mfq::protocol::add_request_runtime_metrics(metrics, state.metrics);
            performance = std::move(metrics);
        }
        Statement write(access.database, "UPDATE responses SET status=?,output_message_id=?,output_json=?,finish_reason=?,usage_json=?,performance_json=?,error_json=?,completed_at=? WHERE id=? AND status='running'");
        write.bind(1, state.status); write.bind(2, message_id); write.bind(3, output.dump()); write.bind(4, state.resource["finish_reason"]);
        write.bind(5, usage.is_null() ? usage : Json(usage.dump())); write.bind(6, performance.is_null() ? performance : Json(performance.dump()));
        write.bind(7, state.error.is_null() ? state.error : Json(state.error.dump())); write.bind(8, timestamp()); write.bind(9, state.resource.at("id")); write.step();
        Statement update(access.database, "UPDATE sessions SET state=?,updated_at=? WHERE id=?");
        update.bind(1, state.status == "completed" ? "idle" : state.status == "cancelled" ? "interrupted" : "error");
        update.bind(2, timestamp()); update.bind(3, id); update.step();
        auto saved = response(access.database, state.resource.at("id"));
        transaction.commit(); state.resource = std::move(saved);
    } catch (const std::exception&) {
        throw Error(Error::save_failed, "response_save_failed", "reply ended but persistence failed",
            {{"response_id", state.resource.at("id")}, {"execution_status", state.execution}, {"saved", false}});
    }
}
void Sessions::release(const std::string& id, const Reply::State* state) {
    std::lock_guard guard(mutex_);
    const auto found = active_.find(id);
    if (found == active_.end()) return;
    const auto current = found->second.lock();
    // A new reply may start after the old transaction commits but before this cleanup.
    if (!current || current->state_.get() == state) active_.erase(found);
}
Sessions::Json Sessions::cancel(const std::string& id) {
    std::shared_ptr<Reply> reply;
    { std::lock_guard guard(mutex_); const auto found = active_.find(id); if (found != active_.end()) reply = found->second.lock(); }
    if (!reply) { get(id); throw Error(Error::conflict, "response_not_in_progress", "session has no active reply"); }
    reply->cancel(); return reply->snapshot();
}
void Sessions::request_stop() {
    std::vector<std::shared_ptr<Reply>> replies;
    { std::lock_guard guard(mutex_); stopping_ = true; for (const auto& item : active_) if (auto reply = item.second.lock()) replies.push_back(std::move(reply)); }
    std::exception_ptr failure;
    for (const auto& reply : replies) try { reply->cancel(); } catch (...) { if (!failure) failure = std::current_exception(); }
    if (failure) std::rethrow_exception(failure);
}
void Sessions::shutdown() {
    std::exception_ptr failure;
    try { request_stop(); } catch (...) { failure = std::current_exception(); }
    std::vector<std::shared_ptr<Reply>> replies;
    { std::lock_guard guard(mutex_); for (const auto& item : active_) if (auto reply = item.second.lock()) replies.push_back(std::move(reply)); }
    for (const auto& reply : replies) try { reply->finish(); } catch (...) { if (!failure) failure = std::current_exception(); }
    if (failure) std::rethrow_exception(failure);
}
} // namespace mfq::server
