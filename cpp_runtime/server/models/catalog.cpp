#include "catalog.h"
#include "errors.h"
#include "instance.h"
#include "storage/files.h"
#include "tyloinfer/gguf_model_source.h"
#include "tyloinfer/model_source.h"
#include "mfq_paged_prefix_cache.h"

#include <algorithm>
#include <cctype>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <nlohmann/json.hpp>

namespace mfq::server {
namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;
constexpr std::size_t max_entries = 100000;
constexpr std::size_t max_index_bytes = 4 * 1024 * 1024;

/** Reports an explicitly selected directory that is missing or inaccessible. */
[[noreturn]] void unavailable_directory() {
    throw ModelError(ModelError::Kind::not_found, "model_directory_not_found", "model directory is unavailable");
}

std::string hash(const std::string& value) {
    return mfq::cache::block_hash_hex(mfq::cache::sha256(value)).substr(0, 32);
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

// TQ and MFQ share the MFQ1 container; GGUF checkpoints are separate.
bool is_container(const fs::path& path) {
    const auto extension = lowercase(path_utf8(path.extension()));
    return extension == ".tq" || extension == ".mfq";
}

bool is_gguf_name(const fs::path& path) {
    const auto name = lowercase(path_utf8(path.filename()));
    return name.size() >= 5 && name.compare(name.size() - 5, 5, ".gguf") == 0;
}

bool is_gguf_part(const fs::path& path) {
    return lowercase(path_utf8(path.filename())).find(".gguf.part-") != std::string::npos;
}

bool is_model_file(const fs::path& path) {
    return is_container(path) || is_gguf_name(path) || is_gguf_part(path);
}

// Leave unreadable GGUFs visible so inspect() can report their errors.
bool is_gguf_table(const fs::path& path) {
    if (!is_gguf_name(path) && !is_gguf_part(path)) return false;
    try { return mfq::gguf_is_table_component(path); } catch (...) { return false; }
}

bool is_hf(const fs::path& path) {
    std::error_code error;
    if (!fs::is_regular_file(path / "config.json", error)) return false;
    if (fs::is_regular_file(path / "model.safetensors.index.json", error)) return true;
    for (auto it = fs::directory_iterator(path, fs::directory_options::skip_permission_denied, error);
            it != fs::directory_iterator(); it.increment(error)) {
        std::error_code item_error;
        if (it->is_regular_file(item_error) && it->path().extension() == ".safetensors") return true;
    }
    return false;
}

fs::path model_key(const fs::path& path) {
    if (!is_model_file(path)) return path;
    static const std::regex shard(R"(^(.*)-([0-9]{5})-of-([0-9]{5})$)");
    static const std::regex part(R"(^(.*\.gguf)\.part-([0-9]{1,5})-of-([0-9]{1,5})$)");
    std::smatch match;
    const auto filename = path_utf8(path.filename());
    if (std::regex_match(filename, match, part)) return path.parent_path() / fs::u8path(match[1].str());
    const auto stem = path_utf8(path.stem());
    return std::regex_match(stem, match, shard)
        ? path.parent_path() / fs::u8path(match[1].str() + lowercase(path_utf8(path.extension())))
        : path;
}

std::set<fs::path> discover(const fs::path& root) {
    std::set<fs::path> paths;
    for (const auto& part : root) if (part == ".tyloinfer-downloads") return paths;
    std::error_code error;
    if (!fs::is_directory(root, error)) return paths;
    if (is_hf(root)) paths.insert(root);
    std::size_t visited = 0;
    for (auto it = fs::recursive_directory_iterator(root, fs::directory_options::skip_permission_denied, error);
            it != fs::recursive_directory_iterator(); it.increment(error)) {
        if (error) throw ModelError(ModelError::Kind::failure, "model_scan_failed", "could not scan model directory");
        if (it->path().filename() == ".tyloinfer-downloads") { it.disable_recursion_pending(); continue; }
        if (++visited > max_entries || it.depth() > 32)
            throw ModelError(ModelError::Kind::too_large, "model_catalog_too_large", "model directory exceeds scan limits");
        if (!it->is_regular_file(error)) continue;
        const auto& path = it->path();
        if (is_model_file(path) && !is_gguf_table(path)) paths.insert(fs::canonical(path));
        else if (path.filename() == "config.json" && is_hf(path.parent_path())) paths.insert(fs::canonical(path.parent_path()));
    }
    return paths;
}

std::string modified_at(fs::file_time_type value) {
    const auto system = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        value - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
    const auto seconds = std::chrono::system_clock::to_time_t(system);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &seconds);
#else
    gmtime_r(&seconds, &utc);
#endif
    std::ostringstream result;
    result << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return result.str();
}

std::optional<std::int64_t> context_capacity(const json& config) {
    for (const auto* source : {&config, config.contains("text_config") ? &config["text_config"] : &config,
            config.contains("language_config") ? &config["language_config"] : &config}) {
        if (!source->is_object()) continue;
        for (const auto* key : {"max_position_embeddings", "max_seq_len", "max_seq_length", "seq_length", "n_positions"}) {
            const auto found = source->find(key);
            if (found != source->end() && found->is_number_integer() && *found > 0 && *found <= 2147483647)
                return found->get<std::int64_t>();
        }
    }
    return {};
}

ModelCatalog::Artifact inspect(const fs::path& path, const std::vector<fs::path>& group) {
    const bool hf = fs::is_directory(path);
    const auto name = hf ? path_utf8(path.filename()) : path_utf8(model_key(path).stem());
    ModelCatalog::Artifact result;
    result.path = path;
    result.id = hash(path_utf8(path));
    result.name = name;
    result.format = hf ? "hf"
        : lowercase(path_utf8(model_key(path).extension())) == ".gguf" ? "gguf" : "tq";
    result.shard_count = group.size();
    result.modified_at = modified_at(fs::last_write_time(path));
    if (!hf) {
        for (const auto& file : group) result.total_bytes += fs::file_size(file);
        static const std::regex shard(R"(^.*-([0-9]{5})-of-([0-9]{5})(\.gguf|\.tq|\.mfq)$)");
        static const std::regex part(R"(^.*\.gguf\.part-([0-9]{1,5})-of-([0-9]{1,5})$)");
        std::smatch match;
        const auto filename = lowercase(path_utf8(path.filename()));
        if (std::regex_match(filename, match, part) || std::regex_match(filename, match, shard)) {
            const auto expected = std::stoul(match[2]);
            result.shard_count = expected;
            result.missing_shards = expected > group.size() ? expected - group.size() : 0;
        }
    }
    try {
        const std::shared_ptr<const mfq::ModelSource> source = mfq::open_model_source(path);
        auto architecture = std::string(source->architecture());
        result.architecture = architecture.empty() ? "unknown" : architecture;
        constexpr std::string_view suffix = "-hf-full-mfq";
        if (hf && architecture.size() >= suffix.size()) architecture.resize(architecture.size() - suffix.size());
        std::string fingerprint = architecture + '\0' + name;
        std::uint64_t bytes = 0;
        auto latest = fs::file_time_type::min();
        for (const auto& file : source->source_paths()) {
            const auto size = fs::file_size(file);
            bytes += size;
            latest = std::max(latest, fs::last_write_time(file));
            fingerprint += '\0' + path_utf8(file.filename()) + ':' + std::to_string(size);
        }
        std::set<std::string> dtypes;
        for (const auto& tensor : source->tensors()) dtypes.insert(tensor.stored_dtype.empty() ? tensor.dtype : tensor.stored_dtype);
        if (!hf && !source->assets().empty()) dtypes.insert("BLOB");
        const auto config = source->has_asset(mfq::kModelConfigAsset) ? json::parse(source->model_config_json()) : json::object();
        if (!config.is_object()) throw std::runtime_error("model configuration must be an object");
        result.id = hash(fingerprint);
        result.total_bytes = bytes;
        result.shard_count = source->source_paths().size();
        result.missing_shards = 0;
        result.tensor_count = source->tensors().size();
        result.record_count = result.tensor_count + (hf ? 1 : source->assets().size());
        result.dtypes.assign(dtypes.begin(), dtypes.end());
        result.context_capacity = context_capacity(config);
        result.modified_at = modified_at(latest);
        result.complete = true;
    } catch (const std::exception& error) {
        result.error = "model metadata is invalid or incomplete";
        std::cerr << json{{"event", "model_inspection_failed"}, {"model", name}, {"detail", error.what()}}.dump() << '\n';
    }
    return result;
}

bool valid_id(const std::string& id) {
    return id.size() == 32 && std::all_of(id.begin(), id.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

} // namespace

ModelCatalog::ModelCatalog(std::vector<fs::path> roots) : secret_(normalize_uuid({})) {
    for (const auto& root : roots) {
        const auto path = resolved_path(root);
        if (std::find(roots_.begin(), roots_.end(), path) == roots_.end()) roots_.push_back(path);
        if (fs::exists(path) && !fs::is_directory(path)) throw std::invalid_argument("--model-dir must be a directory");
    }
    fs::create_directories(roots_.at(0));
}

std::set<fs::path> ModelCatalog::registered_paths() const {
    std::set<fs::path> result;
    for (const auto& root : roots_) {
        const auto index = root / ".tq-files.json";
        if (!fs::exists(index)) continue;
        try {
            if (fs::file_size(index) > max_index_bytes) throw std::runtime_error("index too large");
            std::ifstream stream(index);
            const auto document = json::parse(stream);
            if (!document.is_object() || document.value("version", 0) != 1 || !document.at("files").is_array() || document["files"].size() > 10000)
                throw std::runtime_error("invalid index");
            for (const auto& item : document["files"]) {
                if (!item.is_string()) throw std::runtime_error("invalid index path");
                const auto value = item.get<std::string>();
                if (value.empty() || value.size() > 4096 || value.find('\0') != std::string::npos)
                    throw std::runtime_error("invalid index path");
                const auto path = fs::u8path(value);
                if (!path.is_absolute()) throw std::runtime_error("index paths must be absolute");
                result.insert(path);
            }
        } catch (...) { throw ModelError(ModelError::Kind::failure, "model_catalog_index_invalid", "model catalog index is invalid; existing file was preserved"); }
    }
    return result;
}

std::vector<ModelCatalog::Artifact> ModelCatalog::scan(const std::set<fs::path>& registered) const {
    auto paths = registered;
    for (const auto& root : roots_) {
        const auto found = discover(root);
        paths.insert(found.begin(), found.end());
    }
    std::map<fs::path, std::vector<fs::path>> groups;
    for (const auto& path : paths) {
        std::error_code error;
        if ((fs::is_directory(path, error) && is_hf(path)) || (is_model_file(path) && fs::is_regular_file(path, error))) {
            const auto canonical = fs::canonical(path);
            auto& group = groups[model_key(canonical)];
            if (std::find(group.begin(), group.end(), canonical) == group.end()) group.push_back(canonical);
        }
    }
    std::map<std::string, Artifact> names;
    std::set<std::pair<std::string, std::string>> representations;
    for (auto& [key, group] : groups) {
        std::sort(group.begin(), group.end());
        // Shards and raw parts open through the group key when it is present.
        std::error_code error;
        const auto entry = fs::is_regular_file(key, error) ? key : group.front();
        Artifact artifact = inspect(entry, group);
        const auto name = artifact.name;
        if (!representations.emplace(name, artifact.format).second)
            throw ModelError(ModelError::Kind::conflict, "duplicate_model_name", "multiple model artifacts have the same name");
        const auto previous = names.find(name);
        if (previous != names.end()) {
            if (artifact.format == "hf") continue;
        }
        names.insert_or_assign(name, std::move(artifact));
    }
    std::vector<Artifact> result;
    for (auto& [name, artifact] : names) result.push_back(std::move(artifact));
    return result;
}

void ModelCatalog::refresh() {
    auto next = scan(registered_paths());
    artifacts_ = std::move(next);
    scanned_ = true;
    scanned_at_ = std::chrono::steady_clock::now();
}

std::vector<ModelCatalog::Artifact> ModelCatalog::list(bool requested) {
    std::lock_guard<std::mutex> guard(mutex_);
    // shortcut: refresh scans on HTTP workers; move scans off workers when slow volumes require it.
    if (requested || !scanned_ || std::chrono::steady_clock::now() - scanned_at_ >= std::chrono::seconds(5)) refresh();
    return artifacts_;
}

std::string ModelCatalog::directory_id(const fs::path& path) {
    const auto id = hash(secret_ + path_utf8(path));
    if (directories_.size() >= max_entries && !directories_.count(id))
        throw ModelError(ModelError::Kind::too_large, "model_catalog_too_large", "too many browsed model directories");
    directories_[id] = path;
    return id;
}

fs::path ModelCatalog::directory(const DirectorySource& source) const {
    const auto& id = source.id;
    const auto path = path_utf8(source.path);
    if (!id.empty() && !valid_id(id)) throw ModelError(ModelError::Kind::invalid, "invalid_model_directory_source", "invalid directory identifier");
    if (!id.empty() && !path.empty()) throw ModelError(ModelError::Kind::invalid, "invalid_model_directory_source", "directory_id and path cannot be used together");
    if (path.size() > 4096 || path.find('\0') != std::string::npos)
        throw ModelError(ModelError::Kind::invalid, "invalid_model_directory_source", "invalid directory path");
    fs::path selected;
    if (!path.empty()) {
        try { selected = resolved_path(source.path); }
        catch (const fs::filesystem_error&) { unavailable_directory(); }
    }
    else {
        const auto found = directories_.find(id);
        if (found != directories_.end()) selected = found->second;
    }
    std::error_code error;
    if (selected.empty() || !fs::is_directory(selected, error))
        unavailable_directory();
    const auto canonical = fs::canonical(selected, error);
    if (error) unavailable_directory();
    return canonical;
}

std::optional<ModelCatalog::DirectoryEntry> ModelCatalog::directory_entry(const fs::path& path) {
    std::error_code error;
    auto it = fs::directory_iterator(path, error);
    if (error) return {};
    std::set<fs::path> models;
    if (is_hf(path)) models.insert(path);
    std::size_t visited = 0;
    for (; it != fs::directory_iterator(); it.increment(error)) {
        if (++visited > max_entries) throw ModelError(ModelError::Kind::too_large, "model_catalog_too_large", "directory exceeds scan limits");
        std::error_code item_error;
        if (it->is_regular_file(item_error) && is_model_file(it->path()) && !is_gguf_table(it->path()))
            models.insert(model_key(it->path()));
        else if (it->is_directory(item_error) && is_hf(it->path())) models.insert(it->path());
    }
    if (error) return {};
    auto name = path_utf8(path.filename());
    if (name.empty()) name = path_utf8(path.root_path());
    return DirectoryEntry{directory_id(path), name, models.size()};
}

ModelCatalog::DirectoryListing ModelCatalog::browse(const fs::path& path) {
    DirectoryListing result;
    result.current = directory_entry(path);
    if (!result.current) unavailable_directory();
    result.path = path;
    if (path.parent_path() != path) result.parent_id = directory_id(path.parent_path());
    std::map<std::string, fs::path> children;
    std::error_code error;
    auto it = fs::directory_iterator(path, error);
    if (error) unavailable_directory();
    for (; it != fs::directory_iterator(); it.increment(error)) {
        const auto& entry = *it;
        if (entry.path().filename() == ".tyloinfer-downloads") continue;
        std::error_code item_error;
        if (entry.is_directory(item_error)) {
            const auto child = fs::canonical(entry.path(), item_error);
            if (!item_error) children.emplace(path_utf8(entry.path().filename()), child);
        } else if (entry.is_regular_file(item_error) && is_model_file(entry.path())) {
            const auto size = entry.file_size(item_error);
            if (!item_error) result.files.push_back({path_utf8(entry.path().filename()), size});
        }
    }
    if (error) unavailable_directory();
    for (const auto& [name, child] : children)
        if (auto entry = directory_entry(child)) result.children.push_back(std::move(*entry));
    std::sort(result.files.begin(), result.files.end(), [](const DirectoryFile& a, const DirectoryFile& b) { return a.name < b.name; });
    return result;
}

ModelCatalog::DirectoryListing ModelCatalog::browse(const DirectorySource& source) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!source.id.empty() || !source.path.empty()) return browse(directory(source));
    DirectoryListing result;
    for (const auto& root : roots_) {
        std::error_code error;
        if (fs::is_directory(root, error))
            if (auto entry = directory_entry(root)) result.children.push_back(std::move(*entry));
    }
    return result;
}

std::vector<ModelCatalog::Artifact> ModelCatalog::register_directory(const DirectorySource& source) {
    if (source.id.empty() == source.path.empty()) throw ModelError(ModelError::Kind::invalid, "invalid_model_directory_source", "exactly one directory source is required");
    std::lock_guard<std::mutex> guard(mutex_);
    const auto selected = directory(source);
    const auto added = discover(selected);
    if (added.empty()) throw ModelError(ModelError::Kind::invalid, "model_registration_failed", "selected directory contains no supported models");
    auto paths = registered_paths();
    paths.insert(added.begin(), added.end());
    if (paths.size() > 10000) throw ModelError(ModelError::Kind::too_large, "model_catalog_too_large", "too many registered model files");
    auto files = json::array();
    for (const auto& file : paths) files.push_back(path_utf8(file));
    const json document{{"version", 1}, {"files", std::move(files)}};
    if (document.dump().size() > max_index_bytes)
        throw ModelError(ModelError::Kind::too_large, "model_catalog_too_large", "model catalog index exceeds size limit");
    auto next = scan(paths);
    try { write_json(roots_.front() / ".tq-files.json", document); }
    catch (...) { throw ModelError(ModelError::Kind::failure, "model_registration_save_failed", "could not persist model registration"); }
    artifacts_ = std::move(next);
    scanned_ = true;
    scanned_at_ = std::chrono::steady_clock::now();
    std::vector<Artifact> data;
    for (const auto& artifact : artifacts_)
        if (std::mismatch(selected.begin(), selected.end(), artifact.path.begin(), artifact.path.end()).first == selected.end())
            data.push_back(artifact);
    return data;
}

ModelCatalog::DirectoryListing ModelCatalog::model_directory(const std::string& model_id) {
    if (!valid_id(model_id)) throw ModelError(ModelError::Kind::invalid, "invalid_model_id", "invalid model identifier");
    std::lock_guard<std::mutex> guard(mutex_);
    if (!scanned_) refresh();
    auto find = [&] { return std::find_if(artifacts_.begin(), artifacts_.end(), [&](const Artifact& item) { return item.id == model_id; }); };
    auto found = find();
    if (found == artifacts_.end()) { refresh(); found = find(); }
    if (found == artifacts_.end()) throw ModelError(ModelError::Kind::not_found, "model_artifact_not_found", "model artifact is unavailable");
    const auto path = found->format == "hf" ? found->path : found->path.parent_path();
    return browse(directory({{}, path}));
}

ModelCatalog::Artifact ModelCatalog::resolve(const std::string& model) {
    if (model.empty() || model.size() > 255 || model.find('\0') != std::string::npos)
        throw ModelError(ModelError::Kind::invalid, "invalid_model_id", "model must contain 1 to 255 bytes");
    std::lock_guard<std::mutex> guard(mutex_);
    if (!scanned_) refresh();
    const auto find = [&] { return std::find_if(artifacts_.begin(), artifacts_.end(), [&](const Artifact& item) {
        return item.id == model || item.name == model;
    }); };
    auto found = find();
    if (found == artifacts_.end()) { refresh(); found = find(); }
    if (found == artifacts_.end()) throw ModelError(ModelError::Kind::not_found, "model_artifact_not_found", "model artifact is unavailable");
    return *found;
}

} // namespace mfq::server
