#include "instances.h"
#include "errors.h"
#include "storage/storage.h"
#include "storage/sqlite.h"

#include <algorithm>
#include <set>

namespace mfq::server {
namespace {
ModelInstances::Aliases normalize(ModelInstances::Aliases values) {
    std::set<std::string> used;
    for (auto it = values.begin(); it != values.end();) {
        auto& [name, alias] = *it;
        const auto first = alias.find_first_not_of(" \t\r\n\v\f");
        alias = first == std::string::npos ? "" : alias.substr(first, alias.find_last_not_of(" \t\r\n\v\f") - first + 1);
        if (alias.empty() || alias == name) { it = values.erase(it); continue; }
        const auto length = std::count_if(alias.begin(), alias.end(), [](unsigned char c) { return (c & 0xc0) != 0x80; });
        if (length > 255 || std::any_of(alias.begin(), alias.end(), [](unsigned char c) { return c < 32 || c == 127; }))
            throw ModelError(ModelError::Kind::invalid, "invalid_model_alias", "model aliases must be at most 255 characters without ASCII control characters");
        if (!used.insert(alias).second)
            throw ModelError(ModelError::Kind::conflict, "model_alias_conflict", "model aliases must be unique");
        ++it;
    }
    for (const auto& [name, alias] : values) if (values.count(alias))
        throw ModelError(ModelError::Kind::conflict, "model_alias_conflict", "aliases cannot shadow model names");
    return values;
}
}

ModelInstances::Aliases ModelInstances::read_aliases() try {
    auto access = storage_.access();
    sqlite::Statement query(access.database, "SELECT value FROM schema_meta WHERE key='runtime_model_aliases'");
    if (!query.step()) return {};
    return normalize(nlohmann::json::parse(query.column(0).get<std::string>()).get<Aliases>());
} catch (const nlohmann::json::exception&) {
    throw ModelError(ModelError::Kind::failure, "invalid_saved_aliases", "saved model aliases must be an object of strings");
}

ModelInstances::Aliases ModelInstances::aliases() const {
    std::lock_guard guard(mutex_);
    return aliases_;
}

ModelInstances::Aliases ModelInstances::set_aliases(Aliases values) {
    values = normalize(std::move(values));
    std::string encoded;
    try { encoded = nlohmann::json(values).dump(); }
    catch (const nlohmann::json::exception&) {
        throw ModelError(ModelError::Kind::invalid, "invalid_model_alias", "model aliases must be valid UTF-8");
    }
    std::unique_lock operation(operations_, std::try_to_lock);
    if (!operation.owns_lock()) throw ModelError(ModelError::Kind::conflict, "model_operation_busy", "another model operation is in progress");
    if (stopping_) throw ModelError(ModelError::Kind::stopping, "server_stopping", "server is stopping");
    std::map<std::string, std::set<std::string>> names;
    std::set<std::string> ids;
    for (const auto& artifact : catalog_.list(false)) { names[artifact.name].insert(artifact.id); ids.insert(artifact.id); }
    // Sessions acquire Storage before get(); keep that lock order here too.
    auto access = storage_.access();
    std::lock_guard guard(mutex_);
    for (const auto& [id, instance] : instances_) {
        names[instance->model].insert(instance->artifact_id); ids.insert(id); ids.insert(instance->artifact_id);
    }
    for (const auto& [name, alias] : values) {
        if (!names.count(name)) throw ModelError(ModelError::Kind::invalid, "unknown_model", "alias target must be a registered or loaded model");
        if (names.at(name).size() != 1 || names.count(alias) || ids.count(alias))
            throw ModelError(ModelError::Kind::conflict, "model_alias_conflict", "aliases require unique model names and cannot shadow names or IDs");
    }
    auto result = values;
    try {
        sqlite::Transaction transaction(access.database);
        sqlite::Statement write(access.database, "INSERT INTO schema_meta(key,value) VALUES('runtime_model_aliases',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value");
        write.bind(1, encoded); write.step(); transaction.commit();
    } catch (const sqlite::Error&) {
        throw ModelError(ModelError::Kind::failure, "model_alias_save_failed", "model aliases could not be saved");
    }
    aliases_.swap(values);
    return result;
}

std::vector<std::string> ModelInstances::names() const {
    struct Name { std::string model, id; };
    std::map<std::string, Name> candidates;
    for (const auto& artifact : catalog_.list(false))
        if (support(artifact).supported) candidates.emplace(artifact.id, Name{artifact.name, artifact.id});
    std::lock_guard guard(mutex_);
    std::map<std::string, std::size_t> counts;
    std::set<std::string> ids;
    for (const auto& [id, instance] : instances_) {
        if (instance->snapshot().state != InstanceInfo::State::loaded) candidates.erase(instance->artifact_id);
        else candidates.insert_or_assign(instance->artifact_id, Name{instance->model, id});
    }
    for (auto& [artifact, name] : candidates) {
        if (const auto alias = aliases_.find(name.model); alias != aliases_.end()) name.model = alias->second;
        ++counts[name.model]; ids.insert(artifact); ids.insert(name.id);
    }
    std::vector<std::string> result;
    for (const auto& [artifact, name] : candidates)
        result.push_back(counts[name.model] == 1 && !ids.count(name.model) ? name.model : name.id);
    std::sort(result.begin(), result.end());
    return result;
}
}
