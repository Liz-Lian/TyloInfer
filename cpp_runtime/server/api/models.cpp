#include "models.h"
#include "http.h"
#include "protocol.h"
#include "models/instances.h"
#include "models/errors.h"
#include "jobs/jobs.h"
#include "instance.h"

namespace mfq::server {
namespace {
using json = nlohmann::json;

json directory_json(const ModelCatalog::DirectoryListing& listing) {
    auto data = json::array(), files = json::array();
    for (const auto& child : listing.children)
        data.push_back({{"id", child.id}, {"name", child.name}, {"model_file_count", child.model_file_count}});
    for (const auto& file : listing.files) files.push_back({{"name", file.name}, {"byte_size", file.bytes}});
    return {{"current_id", listing.current ? json(listing.current->id) : json(nullptr)},
        {"current_name", listing.current ? json(listing.current->name) : json(nullptr)},
        {"current_path", listing.current ? json(path_utf8(listing.path)) : json(nullptr)},
        {"parent_id", listing.parent_id ? json(*listing.parent_id) : json(nullptr)},
        {"model_file_count", listing.current ? listing.current->model_file_count : 0},
        {"can_open_directory", false}, {"can_open_in_finder", false}, {"files", std::move(files)}, {"data", std::move(data)}};
}

json artifacts(const std::vector<ModelCatalog::Artifact>& entries, const ModelInstances& instances) {
    auto data = json::array();
    for (const auto& item : entries) {
        const auto support = instances.support(item);
        data.push_back({{"id", item.id}, {"name", item.name}, {"architecture", item.architecture}, {"format", item.format},
            {"shard_count", item.shard_count}, {"missing_shards", item.missing_shards}, {"total_bytes", item.total_bytes},
            {"tensor_count", item.tensor_count}, {"record_count", item.record_count}, {"dtypes", item.dtypes},
            {"context_capacity", item.context_capacity ? json(*item.context_capacity) : json(nullptr)},
            {"complete", item.complete}, {"error", item.error.empty() ? json(nullptr) : json(item.error)},
            {"loadable", support.supported}, {"support_error", support.reason.empty() ? json(nullptr) : json(support.reason)},
            {"modified_at", item.modified_at}});
    }
    return {{"data", std::move(data)}};
}

json instance_json(const InstanceInfo& item) {
    const auto state = item.state == InstanceInfo::State::loaded ? "loaded" :
        item.state == InstanceInfo::State::unloading ? "unloading" : "failed";
    return {{"instance_id", item.id}, {"model", item.model}, {"artifact_id", item.artifact_id}, {"backend", item.backend},
        {"state", state}, {"context_size", item.capabilities.max_context}, {"max_requests", item.capabilities.max_requests},
        {"active_requests", item.execution.requests}, {"input_bytes", item.execution.input_bytes},
        {"capabilities", mfq::protocol::model_capability_profile_json(item.capabilities.capabilities)},
        {"policy", {{"pin", item.policy.pin}, {"idle_ttl_seconds", item.policy.idle_ttl ? json(item.policy.idle_ttl->count()) : json(nullptr)}}},
        {"metrics_age_ms", std::max<std::int64_t>(0, std::chrono::duration_cast<std::chrono::milliseconds>(
            engine::Clock::now() - item.execution.captured).count())},
        {"metrics", item.execution.runtime}, {"session_metrics", item.execution.session}};
}

void fields(const json& request, std::initializer_list<const char*> allowed) {
    if (!request.is_object()) throw ApiError(400, "invalid_request", "request must be a JSON object");
    for (const auto& item : request.items())
        if (std::find(allowed.begin(), allowed.end(), item.key()) == allowed.end())
            throw ApiError(400, "invalid_request", "unknown model operation parameter: " + item.key());
}

std::string selector(const json& request, const char* key) {
    if (!request.contains(key) || !request[key].is_string())
        throw ApiError(400, "invalid_request", std::string(key) + " must be a string");
    return request[key].get<std::string>();
}

std::int64_t integer(const json& request, const char* field, std::int64_t fallback, std::int64_t minimum, std::int64_t maximum) {
    if (!request.contains(field)) return fallback;
    const auto& value = request[field];
    if (!value.is_number_integer() || value < minimum || value > maximum)
        throw ApiError(400, "invalid_request", std::string(field) + " is outside its supported range");
    return value.get<std::int64_t>();
}
}

void register_models(httplib::Server& http, ModelCatalog& catalog, ModelInstances& instances, Jobs& jobs) {
    http.Get("/api/v1/runtime/model-aliases", [&](const httplib::Request&, httplib::Response& response) {
        set_json(response, {{"aliases", instances.aliases()}});
    });
    http.Put("/api/v1/runtime/model-aliases", [&](const httplib::Request& request, httplib::Response& response) {
        const auto body = parse_json(request);
        fields(body, {"aliases"});
        if (!body.contains("aliases") || !body["aliases"].is_object())
            throw ApiError(400, "invalid_request", "aliases must be an object of strings");
        for (const auto& value : body["aliases"])
            if (!value.is_string()) throw ApiError(400, "invalid_request", "aliases must be an object of strings");
        set_json(response, {{"aliases", instances.set_aliases(body["aliases"].get<ModelInstances::Aliases>())}});
    });
    http.Get("/api/v1/models", [&](const httplib::Request& request, httplib::Response& response) {
        const auto refresh = request.get_param_value("refresh");
        if (request.has_param("refresh") && refresh != "true" && refresh != "false" && refresh != "1" && refresh != "0")
            throw ApiError(400, "invalid_request", "refresh must be a boolean");
        set_json(response, artifacts(catalog.list(refresh == "true" || refresh == "1"), instances));
    });
    http.Get("/api/v1/models/directories", [&](const httplib::Request& request, httplib::Response& response) {
        const auto id = request.get_param_value("directory_id"), path = request.get_param_value("path");
        if ((request.has_param("directory_id") && id.empty()) || (request.has_param("path") && path.empty()))
            throw ApiError(400, "invalid_model_directory_source", "directory source cannot be empty");
        set_json(response, directory_json(catalog.browse({id, std::filesystem::u8path(path)})));
    });
    http.Post("/api/v1/models/directories/register", [&](const httplib::Request& request, httplib::Response& response) {
        const auto body = parse_json(request);
        fields(body, {"path", "directory_id"});
        for (const auto& item : body.items())
            if (!item.value().is_null() && (!item.value().is_string() || item.value().get_ref<const std::string&>().empty()))
                throw ApiError(400, "invalid_request", "invalid model directory source");
        const auto id = body.contains("directory_id") && !body["directory_id"].is_null() ? body["directory_id"].get<std::string>() : "";
        const auto path = body.contains("path") && !body["path"].is_null() ? body["path"].get<std::string>() : "";
        set_json(response, artifacts(catalog.register_directory({id, std::filesystem::u8path(path)}), instances));
    });
    http.Get("/api/v1/models/([^/]+)/directory", [&](const httplib::Request& request, httplib::Response& response) {
        set_json(response, directory_json(catalog.model_directory(request.matches[1])));
    });
    http.Get("/api/v1/models/([^/]+)/capabilities", [&](const httplib::Request& request, httplib::Response& response) {
        const auto artifact = catalog.resolve(request.matches[1]);
        if (artifact.id != request.matches[1]) throw ApiError(400, "invalid_request", "capability lookup requires an artifact ID");
        const auto support = instances.support(artifact);
        set_json(response, {{"artifact_id", artifact.id}, {"supported", support.supported},
            {"reason", support.reason.empty() ? json(nullptr) : json(support.reason)},
            {"declared_context_capacity", artifact.context_capacity ? json(*artifact.context_capacity) : json(nullptr)},
            {"load", {{"cancellation", "after_return"}, {"progress", "phase"}}}});
    });
    http.Post("/api/v1/models/load", [&](const httplib::Request& request, httplib::Response& response) {
        const auto body = parse_json(request);
        fields(body, {"artifact_id", "requirements", "policy"});
        const auto artifact_id = selector(body, "artifact_id");
        const auto artifact = catalog.resolve(artifact_id);
        if (artifact.id != artifact_id) throw ApiError(400, "invalid_request", "loading requires an artifact ID");
        const auto requirements = body.value("requirements", json::object());
        fields(requirements, {"context_size"});
        LoadOptions options;
        options.context_size = integer(requirements, "context_size", 0, 1, 2147483647);
        const auto policy_body = body.value("policy", json::object());
        fields(policy_body, {"pin", "idle_ttl_seconds"});
        LoadPolicy policy;
        if (policy_body.contains("pin")) {
            if (!policy_body["pin"].is_boolean()) throw ApiError(400, "invalid_request", "pin must be a boolean");
            policy.pin = policy_body["pin"].get<bool>();
        }
        if (policy_body.contains("idle_ttl_seconds"))
            policy.idle_ttl = std::chrono::seconds(integer(policy_body, "idle_ttl_seconds", 0, 1, 2147483647));
        if (policy.pin && policy.idle_ttl) throw ApiError(400, "invalid_request", "pinned instances cannot have an idle timeout");
        const auto operation = jobs.submit("model.load", body, [&, artifact_id, options, policy](const Jobs::Control& control) {
            const auto progress = [&](LoadControl::Phase phase) {
                control.progress(phase == LoadControl::Phase::resolving ? "resolving" :
                    phase == LoadControl::Phase::loading ? "loading" : "initializing");
            };
            try {
                const auto loaded = instances.load(artifact_id, options, {&control.cancelled, progress}, policy);
                return instance_json(loaded);
            } catch (const ModelError& error) {
                if (error.kind == ModelError::Kind::cancelled) throw Jobs::Cancelled{};
                throw;
            }
        });
        response.set_header("Location", "/api/v1/jobs/" + operation);
        set_json(response, {{"operation_id", operation}, {"status", "accepted"}}, 202);
    });
    http.Post("/api/v1/models/unload", [&](const httplib::Request& request, httplib::Response& response) {
        const auto body = parse_json(request);
        fields(body, {"instance_id", "force"});
        const auto id = normalize_uuid(selector(body, "instance_id"));
        if (body.contains("force") && !body["force"].is_boolean())
            throw ApiError(400, "invalid_request", "force must be a boolean");
        const bool force = body.value("force", false);
        const auto operation = jobs.submit("model.unload", body, [&, id, force](const Jobs::Control& control) {
            control.checkpoint();
            instances.unload(id, force);
            return json{{"instance_id", id}, {"state", "unloaded"}};
        });
        response.set_header("Location", "/api/v1/jobs/" + operation);
        set_json(response, {{"operation_id", operation}, {"status", "accepted"}}, 202);
    });
    http.Get("/api/v1/runtime/instances", [&](const httplib::Request&, httplib::Response& response) {
        auto data = json::array();
        for (const auto& item : instances.list()) data.push_back(instance_json(item));
        set_json(response, {{"data", std::move(data)}});
    });
}
}
