#include "hub.h"
#include "hub/hub.h"
#include "http.h"
#include "instance.h"
#include "jobs/jobs.h"
#include "models/catalog.h"
#include "models/errors.h"
#include "openapi.h"
#include <map>
#include <regex>
#include <tuple>

namespace mfq::server {
namespace {
using Json = nlohmann::json;
void fields(const Json& body, std::initializer_list<const char*> allowed) {
    if (!body.is_object()) throw ApiError(400, "invalid_request", "expected a JSON object");
    for (const auto& item : body.items()) if (std::find(allowed.begin(), allowed.end(), item.key()) == allowed.end())
        throw ApiError(400, "invalid_request", "unknown parameter: " + item.key());
}
std::string text(const Json& body, const char* key, bool required = false) {
    if (!body.contains(key) || body[key].is_null()) {
        if (!required) return {};
    } else if (body[key].is_string() && !body[key].get_ref<const std::string&>().empty() && body[key].get_ref<const std::string&>().size() <= 4096 &&
            body[key].get_ref<const std::string&>().find('\0') == std::string::npos) return body[key].get<std::string>();
    throw ApiError(400, "invalid_request", std::string(key) + " must be a nonempty string");
}
Hub::Provider provider(const std::string& name) {
    if (name == "huggingface") return Hub::Provider::hf;
    if (name == "modelscope") return Hub::Provider::ms;
    throw ApiError(400, "invalid_provider", "provider must be huggingface or modelscope");
}
const char* provider(Hub::Provider name) { return name == Hub::Provider::hf ? "huggingface" : "modelscope"; }
Json directory(const Hub::Directory& value) {
    return {{"directory", value.selected.empty() ? Json(nullptr) : Json(path_utf8(value.selected))},
        {"actual_directory", path_utf8(value.actual)}, {"default_directory", path_utf8(value.fallback)},
        {"disk_total_bytes", value.space ? Json(value.space->capacity) : Json(nullptr)},
        {"disk_free_bytes", value.space ? Json(value.space->free) : Json(nullptr)},
        {"disk_available_bytes", value.space ? Json(value.space->available) : Json(nullptr)}, {"evict_prefix_cache", false}};
}
Json repository(const Hub::Repo& repo) {
    Json files = Json::array(), variants = Json::array();
    std::map<std::string, Json> groups;
    std::uint64_t total = 0;
    static const std::regex shard(R"((.*)-[0-9]{5}-of-[0-9]{5}(\.tq|\.mfq|\.gguf))");
    for (const auto& file : repo.files) {
        total += file.size;
        files.push_back({{"name", file.name}, {"byte_size", file.size}, {"sha256", file.sha256.empty() ? Json(nullptr) : Json(file.sha256)}});
        const auto path = std::filesystem::u8path(file.name);
        const auto extension = path.extension();
        if (extension != ".tq" && extension != ".mfq" && extension != ".safetensors" && extension != ".gguf") continue;
        const auto format = extension == ".gguf" ? "gguf" : extension == ".safetensors" ? "hf" : "tq";
        std::string key = file.name;
        if (extension == ".safetensors") key = path_utf8(path.parent_path() / "checkpoint");
        else { std::smatch match; if (std::regex_match(file.name, match, shard)) key = match[1].str() + match[2].str(); }
        auto& group = groups[key];
        // HF 目录名也是模型名；保留来源，避免多个仓库都被下载为 checkpoint。
        const auto label = extension == ".safetensors" ? std::string(provider(repo.source.provider)) + "/" + repo.source.repo + "/" + key : key;
        if (group.is_null()) group = {{"id", key}, {"label", label}, {"format", format}, {"files", Json::array()}, {"byte_size", 0},
            {"configuration", {{"status", "unknown"}, {"recommendation", "unknown"}, {"reasons", Json::array()}}}};
        group["files"].push_back(file.name); group["byte_size"] = group["byte_size"].get<std::uint64_t>() + file.size;
    }
    for (auto& [name, value] : groups) variants.push_back(std::move(value));
    return {{"provider", provider(repo.source.provider)}, {"repo_id", repo.source.repo}, {"revision", repo.source.revision},
        {"source_url", repo.url}, {"downloads", nullptr}, {"likes", nullptr}, {"total_bytes", total},
        {"files", files}, {"variants", variants}, {"tags", Json::array()}, {"architectures", Json::array()},
        {"modalities", Json::array()}, {"gated", nullptr}, {"runtime_compatible", nullptr}};
}
Hub::Request download(const std::string& kind, Json& body, const Hub& hub) {
    if (kind != "download.huggingface" && kind != "download.modelscope")
        throw ApiError(501, "job_kind_not_supported", "this job kind is not available in the native server");
    fields(body, {"repo_id", "revision", "destination", "download_directory", "include", "exclude", "max_workers", "direct", "expected_bytes", "repo_type"});
    Hub::Request result;
    result.source = {provider(kind.substr(9)), text(body, "repo_id", true), text(body, "revision")};
    result.destination = text(body, "destination", true);
    if (body.contains("repo_type") && body["repo_type"] != "model") throw ApiError(501, "repo_type_not_supported", "only model repositories are supported");
    auto selected = text(body, "download_directory");
    if (selected.empty()) selected = path_utf8(hub.directory().actual);
    result.directory = std::filesystem::u8path(selected); body["download_directory"] = selected;
    for (auto [key, target, limit] : {std::tuple{"include", &result.include, 4096u}, {"exclude", &result.exclude, 64u}}) {
        if (!body.contains(key)) continue;
        if (!body[key].is_array() || body[key].size() > limit) throw ApiError(400, "invalid_request", "invalid file filters");
        for (const auto& value : body[key]) {
            if (!value.is_string() || value.get_ref<const std::string&>().empty() || value.get_ref<const std::string&>().size() > 1024)
                throw ApiError(400, "invalid_request", "file patterns must be nonempty strings");
            target->push_back(value.get<std::string>());
        }
    }
    if (body.contains("max_workers")) {
        if (!body["max_workers"].is_number_integer() || body["max_workers"] < 1 || body["max_workers"] > 16)
            throw ApiError(400, "invalid_request", "max_workers must be between 1 and 16");
        result.workers = body["max_workers"].get<std::size_t>();
    }
    if (body.contains("direct")) {
        if (!body["direct"].is_boolean()) throw ApiError(400, "invalid_request", "direct must be boolean");
        result.direct = body["direct"].get<bool>();
    }
    if (body.contains("expected_bytes") && !body["expected_bytes"].is_null()) {
        if (!body["expected_bytes"].is_number_integer() || body["expected_bytes"] < 0 || body["expected_bytes"] > INT64_MAX)
            throw ApiError(400, "invalid_request", "expected_bytes must be a nonnegative 64-bit integer");
        result.expected_bytes = body["expected_bytes"].get<std::uint64_t>();
    }
    return result;
}
std::string submit(Hub& hub, ModelCatalog& catalog, Jobs& jobs, const std::string& kind, Json body) {
    const auto request = download(kind, body, hub);
    return jobs.submit(kind, body, [&, request](const Jobs::Control& control) {
        control.progress("resolving");
        Hub::Result result;
        try {
            result = hub.download(request, {[&] { return control.cancelled.load(); },
                [&](std::uint64_t bytes, std::uint64_t total, std::size_t files) {
                    control.progress("downloading", total ? static_cast<double>(bytes) / total : 0,
                        {{"downloaded_bytes", bytes}, {"total_bytes", total}, {"files_completed", files}});
                }});
        } catch (const Hub::Error& error) {
            if (error.kind == Hub::Error::cancelled) throw Jobs::Cancelled{};
            throw;
        }
        auto models = Json::array();
        try { for (const auto& artifact : catalog.register_directory({{}, result.path})) models.push_back(artifact.id); }
        catch (const ModelError& error) { if (std::string(error.code) != "model_registration_failed") throw; }
        return Json{{"path", path_utf8(result.path)}, {"repo_id", result.source.repo}, {"revision", result.source.revision},
            {"total_bytes", result.bytes}, {"files_completed", result.files}, {"models", models},
            {"registration", models.empty() ? "unsupported" : "registered"}};
    });
}
}
void register_hub(httplib::Server& http, Hub& hub, ModelCatalog& catalog, Jobs& jobs) {
    http.Get("/api/v1/hub/download-directory", [&](const httplib::Request&, httplib::Response& response) { set_json(response, directory(hub.directory())); });
    http.Put("/api/v1/hub/download-directory", [&](const httplib::Request& request, httplib::Response& response) {
        const auto body = parse_json(request); fields(body, {"directory_id", "evict_prefix_cache"});
        if (body.contains("evict_prefix_cache")) {
            if (!body["evict_prefix_cache"].is_boolean()) throw ApiError(400, "invalid_request", "evict_prefix_cache must be boolean");
            if (body["evict_prefix_cache"].get<bool>()) throw ApiError(501, "cache_reclaim_not_supported", "download cache reclamation is not available");
        }
        if (!body.contains("directory_id")) { set_json(response, directory(hub.directory())); return; }
        const auto id = text(body, "directory_id");
        set_json(response, directory(hub.configure(id.empty() ? std::optional<std::filesystem::path>{} : catalog.browse(ModelCatalog::DirectorySource{id, {}}).path)));
    });
    http.Get("/api/v1/hub/models/([^/]+)/([^/]+)/([^/]+)", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, repository(hub.info({provider(request.matches[1]), request.matches[2].str() + "/" + request.matches[3].str(), request.get_param_value("revision")},
            {request.is_connection_closed, {}})));
    });
    http.Post("/api/v1/hub/resolve", [&](const httplib::Request& request, httplib::Response& response) {
        const auto body = parse_json(request); fields(body, {"reference", "fallback_provider"});
        auto name = text(body, "reference", true), fallback = text(body, "fallback_provider");
        auto selected = provider(fallback.empty() ? "huggingface" : fallback);
        static const std::regex url(R"(https://(huggingface\.co|modelscope\.cn)/(?:models/)?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/?)");
        std::smatch match;
        if (std::regex_match(name, match, url)) { selected = match[1] == "huggingface.co" ? Hub::Provider::hf : Hub::Provider::ms; name = match[2]; }
        set_json(response, repository(hub.info({selected, name, {}}, {request.is_connection_closed, {}})));
    });
    http.Get("/api/v1/jobs/kinds", [](const httplib::Request&, httplib::Response& response) {
        static const auto schema = Json::parse(openapi_document).at("components").at("schemas").at("Download");
        set_json(response, {{"data", {{{"kind", "download.huggingface"}, {"payload_schema", schema}},
            {{"kind", "download.modelscope"}, {"payload_schema", schema}}}}});
    });
    http.Post("/api/v1/jobs", [&](const httplib::Request& request, httplib::Response& response) {
        const auto body = parse_json(request); fields(body, {"kind", "payload"});
        if (!body.contains("payload")) throw ApiError(400, "invalid_request", "payload is required");
        set_json(response, jobs.get(submit(hub, catalog, jobs, text(body, "kind", true), body.at("payload"))), 202);
    });
    http.Post("/api/v1/jobs/([0-9a-f-]{36})/retry", [&](const httplib::Request& request, httplib::Response& response) {
        const auto previous = jobs.get(normalize_uuid(request.matches[1]));
        if (!Jobs::terminal(previous) || previous.at("status") == "succeeded") throw ApiError(409, "job_state_conflict", "only failed, cancelled or interrupted downloads can be retried");
        set_json(response, jobs.get(submit(hub, catalog, jobs, previous.at("kind"), previous.at("payload"))), 202);
    });
}
}
