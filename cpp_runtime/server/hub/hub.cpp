#include "hub.h"
#include "fetch.h"
#include "instance.h"
#include "storage/files.h"
#include "mfq_paged_prefix_cache.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <limits>
#include <regex>
#include <set>
#include <thread>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <stdio.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/syscall.h>
#endif

namespace mfq::server {
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
using Error = Hub::Error;

fs::path relative(const std::string& name, bool pattern = false) {
    if (name.empty() || name.size() > 1024 || name.front() == '/' || name.back() == '/' || name.find("//") != std::string::npos ||
        name.find_first_of("\\:<>|\"") != std::string::npos || (!pattern && name.find_first_of("*?") != std::string::npos) ||
        std::any_of(name.begin(), name.end(), [](unsigned char ch) { return ch < 32 || ch == 127; }))
        throw Error(Error::invalid, "invalid_download_path", "download paths must be relative file names");
    const auto path = fs::u8path(name);
    static const std::regex device(R"((CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?)", std::regex::icase);
    for (const auto& part : path)
        if (part.empty() || part == "." || part == ".." || part == ".tyloinfer-downloads" ||
            part.native().back() == '.' || part.native().back() == ' ' || std::regex_match(path_utf8(part), device))
            throw Error(Error::invalid, "invalid_download_path", "download path contains a reserved component");
    return path;
}
void safe(const fs::path& root, const fs::path& path) {
    auto current = root;
    for (const auto& part : path) {
        current /= part;
        if (fs::is_symlink(fs::symlink_status(current)))
            throw Error(Error::invalid, "unsafe_download_path", "download path contains a symbolic link");
    }
}
Json read(const fs::path& path) {
    if (fs::file_size(path) > 16 * 1024 * 1024) throw Error(Error::io, "download_state_invalid", "download state is too large");
    std::ifstream input(path);
    return Json::parse(input);
}
fs::path absolute(const std::string& value) {
    if (value.empty() || value.size() > 4096 || value.find('\0') != std::string::npos || !fs::u8path(value).is_absolute())
        throw Error(Error::invalid, "invalid_download_directory", "download directory must be an absolute path");
    return fs::weakly_canonical(fs::u8path(value));
}
void source(Hub::Source& value) {
    static const std::regex repo("[A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_-][A-Za-z0-9_.-]*");
    if (value.repo.size() > 256 || !std::regex_match(value.repo, repo))
        throw Error(Error::invalid, "invalid_repository", "repository must be owner/name");
    if (value.revision.empty()) value.revision = value.provider == Hub::Provider::hf ? "main" : "master";
    if (value.revision.size() > 255 || std::any_of(value.revision.begin(), value.revision.end(), [](unsigned char ch) { return ch < 32 || ch == 127; }))
        throw Error(Error::invalid, "invalid_revision", "invalid repository revision");
}
bool matches(const std::string& pattern, const std::string& name) {
    std::size_t p = 0, n = 0, star = std::string::npos, retry = 0;
    while (n < name.size()) {
        if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == name[n])) { ++p; ++n; }
        else if (p < pattern.size() && pattern[p] == '*') { star = p++; retry = n; }
        else if (star != std::string::npos) { p = star + 1; n = ++retry; }
        else return false;
    }
    while (p < pattern.size() && pattern[p] == '*') ++p;
    return p == pattern.size();
}
bool matches(const std::vector<std::string>& patterns, const std::string& name) {
    return std::any_of(patterns.begin(), patterns.end(), [&](const auto& pattern) { return matches(pattern, name); });
}
void patterns(const std::vector<std::string>& values, std::size_t limit) {
    if (values.size() > limit) throw Error(Error::invalid, "invalid_download_filter", "too many file patterns");
    for (const auto& value : values) {
        relative(value, true);
        if (value.find_first_of("[]") != std::string::npos)
            throw Error(Error::invalid, "invalid_download_filter", "file patterns support only * and ? wildcards");
    }
}
void publish(const fs::path& from, const fs::path& to) {
#ifdef _WIN32
    const bool ok = MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_WRITE_THROUGH);
#elif defined(__APPLE__)
    const bool ok = ::renamex_np(from.c_str(), to.c_str(), RENAME_EXCL) == 0;
#else
    const bool ok = ::syscall(SYS_renameat2, AT_FDCWD, from.c_str(), AT_FDCWD, to.c_str(), 1) == 0;
#endif
    if (!ok) throw Error(Error::io, "download_publish_failed", "could not publish download without replacing an existing directory");
}
}

void Hub::Control::checkpoint() const {
    if (cancelled && cancelled()) throw Error(Error::cancelled, "download_cancelled", "download cancelled");
}
Hub::Hub(fs::path workspace, Network network)
    : workspace_(resolved_path(workspace)), fallback_(workspace_ / "models"), roots_{fallback_}, network_(std::move(network)) {
    static const std::regex endpoint(R"(https?://(?:\[[0-9a-fA-F:]+\]|[A-Za-z0-9.-]+)(?::[0-9]+)?)");
    for (auto* remote : {&network_.hf, &network_.ms}) {
        while (!remote->url.empty() && remote->url.back() == '/') remote->url.pop_back();
        if (!std::regex_match(remote->url, endpoint)) throw Error(Error::invalid, "invalid_hub_endpoint", "hub endpoint must be an HTTP(S) origin");
        if (std::any_of(remote->token.begin(), remote->token.end(), [](unsigned char ch) { return ch < 32 || ch == 127; }))
            throw Error(Error::invalid, "invalid_hub_token", "hub token contains invalid characters");
    }
    const auto config = workspace_ / "download-directory.json";
    if (fs::exists(config)) {
        try {
            const auto saved = read(config);
            for (const auto& item : saved.at("directories")) roots_.push_back(absolute(item.get<std::string>()));
            if (!saved.at("selected").is_null()) selected_ = absolute(saved.at("selected").get<std::string>());
            if (!selected_.empty() && std::find(roots_.begin(), roots_.end(), selected_) == roots_.end()) throw std::runtime_error("unregistered directory");
        } catch (...) { throw Error(Error::io, "download_settings_invalid", "saved download directories are invalid; existing file was preserved"); }
    }
}
Hub::Remote Hub::remote(Provider provider) const { return provider == Provider::hf ? network_.hf : network_.ms; }
Hub::Repo Hub::info(Source request, const Control& control, bool direct) const {
    source(request); control.checkpoint();
    const auto endpoint = remote(request.provider);
    const auto repo = hub::escape(request.repo, true), revision = hub::escape(request.revision);
    const auto path = request.provider == Provider::hf ? "/api/models/" + repo + "/revision/" + revision + "?blobs=true" :
        "/api/v1/models/" + repo + "/repo/files?Revision=" + revision + "&Recursive=True";
    try {
        const auto document = Json::parse(hub::get({endpoint, network_.ca, direct}, path, control));
        Repo result{request, endpoint.url + (request.provider == Provider::ms ? "/models/" : "/") + repo, {}};
        Json files;
        if (request.provider == Provider::hf) {
            result.source.revision = document.at("sha").get<std::string>();
            if ((result.source.revision.size() != 40 && result.source.revision.size() != 64) ||
                result.source.revision.find_first_not_of("0123456789abcdef") != std::string::npos) throw std::runtime_error("invalid commit");
            source(result.source);
            files = document.at("siblings");
        } else {
            if (document.value("Code", 200) != 200 || !document.value("Success", document.value("success", true)))
                throw Error(Error::network, "hub_metadata_failed", "repository rejected the file listing");
            const auto& data = document.contains("Data") ? document.at("Data") : document.at("data");
            files = data.contains("Files") ? data.at("Files") : data.at("files");
        }
        if (!files.is_array() || files.size() > 100000) throw std::runtime_error("invalid file listing");
        std::set<std::string> names;
        std::uint64_t total = 0;
        for (const auto& item : files) {
            control.checkpoint();
            if (item.value("Type", item.value("type", "blob")) == "tree") continue;
            File file;
            Json size;
            if (request.provider == Provider::hf) {
                file.name = item.at("rfilename").get<std::string>();
                size = item.contains("size") ? item.at("size") : item.at("lfs").at("size");
                if (item.contains("lfs") && !item["lfs"].is_null()) file.sha256 = item.at("lfs").value("sha256", "");
            } else {
                file.name = item.contains("Path") ? item.at("Path").get<std::string>() : item.at("path").get<std::string>();
                size = item.contains("Size") ? item.at("Size") : item.at("size");
                file.sha256 = item.value("Sha256", item.value("sha256", ""));
            }
            relative(file.name);
            if (!names.insert(file.name).second || !size.is_number_integer() || size < 0 || size > INT64_MAX) throw std::runtime_error("invalid file size or name");
            file.size = size.get<std::uint64_t>();
            if (file.size > INT64_MAX - total) throw std::runtime_error("repository size exceeds limit");
            total += file.size;
            if (!file.sha256.empty() && (file.sha256.size() != 64 || !std::all_of(file.sha256.begin(), file.sha256.end(), [](char ch) {
                return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
            }))) throw std::runtime_error("invalid checksum");
            if (request.provider == Provider::ms && file.sha256.empty()) throw std::runtime_error("missing checksum");
            result.files.push_back(std::move(file));
        }
        std::sort(result.files.begin(), result.files.end(), [](const auto& a, const auto& b) { return a.name < b.name; });
        return result;
    } catch (const Json::exception&) { throw Error(Error::integrity, "invalid_hub_metadata", "repository returned invalid metadata"); }
    catch (const Error&) { throw; }
    catch (const std::runtime_error&) { throw Error(Error::integrity, "invalid_hub_metadata", "repository returned invalid file metadata"); }
}
Hub::Directory Hub::directory() const {
    std::lock_guard lock(mutex_);
    Directory result{selected_, selected_.empty() ? fallback_ : selected_, fallback_, {}};
    std::error_code error; const auto space = fs::space(result.actual, error);
    if (!error) result.space = space;
    return result;
}
Hub::Directory Hub::configure(const std::optional<fs::path>& path) {
    {
        std::lock_guard lock(mutex_);
        auto selected = path ? absolute(path_utf8(*path)) : fs::path{};
        if (!selected.empty() && !fs::is_directory(selected)) throw Error(Error::invalid, "invalid_download_directory", "download directory is unavailable");
        auto roots = roots_;
        if (!selected.empty() && std::find(roots.begin(), roots.end(), selected) == roots.end()) roots.push_back(selected);
        auto directories = Json::array(); for (const auto& root : roots) directories.push_back(path_utf8(root));
        try { write_json(workspace_ / "download-directory.json", {{"selected", selected.empty() ? Json(nullptr) : Json(path_utf8(selected))},
            {"directories", directories}, {"evict_prefix_cache", false}}); }
        catch (...) { throw Error(Error::io, "download_settings_save_failed", "could not save download directory"); }
        selected_ = std::move(selected); roots_ = std::move(roots);
    }
    return directory();
}
Hub::Result Hub::download(const Request& request, const Control& control) try {
    std::unique_lock transfer(transfer_, std::try_to_lock);
    if (!transfer) throw Error(Error::conflict, "download_busy", "another download is running");
    if (request.workers < 1 || request.workers > 16) throw Error(Error::invalid, "invalid_download_workers", "max_workers must be between 1 and 16");
    patterns(request.include, 4096); patterns(request.exclude, 64);
    auto name = request.destination;
    if (name.rfind("models/", 0) == 0) name.erase(0, 7);
    const auto destination = relative(name);
    fs::path root;
    {
        std::lock_guard lock(mutex_);
        root = request.directory ? absolute(path_utf8(*request.directory)) : selected_.empty() ? fallback_ : selected_;
        if (std::find(roots_.begin(), roots_.end(), root) == roots_.end()) throw Error(Error::invalid, "invalid_download_directory", "download directory has not been selected on this server");
    }
    if (root == fallback_) fs::create_directories(root);
    if (!fs::is_directory(root) || fs::is_symlink(root)) throw Error(Error::invalid, "invalid_download_directory", "download directory is unavailable");
    safe(root, destination);
    if (fs::exists(root / destination)) throw Error(Error::conflict, "download_destination_exists", "download destination already exists");
    auto repo = info(request.source, control, request.direct);
    repo.files.erase(std::remove_if(repo.files.begin(), repo.files.end(), [&](const auto& file) {
        return (!request.include.empty() && !matches(request.include, file.name)) || matches(request.exclude, file.name);
    }), repo.files.end());
    if (repo.files.empty()) throw Error(Error::invalid, "empty_download", "file selection is empty");
    std::uint64_t total = 0;
    auto entries = Json::array();
    std::set<std::string> names;
    for (const auto& file : repo.files) {
        auto parent = fs::u8path(file.name).parent_path();
        for (; !parent.empty(); parent = parent.parent_path())
            if (names.count(path_utf8(parent))) throw Error(Error::invalid, "invalid_download_path", "file and directory names overlap");
        names.insert(file.name);
        if (file.size > INT64_MAX - total) throw Error(Error::invalid, "download_too_large", "download exceeds the supported size");
        total += file.size; entries.push_back({{"name", file.name}, {"size", file.size}, {"sha256", file.sha256}});
    }
    const auto key = mfq::cache::block_hash_hex(mfq::cache::sha256(name));
    const auto stage = root / ".tyloinfer-downloads" / key, content = stage / "files", parts = stage / "parts";
    safe(root, content.lexically_relative(root)); safe(root, parts.lexically_relative(root));
    fs::create_directories(content); fs::create_directories(parts);
    const auto manifest = stage / "source.json";
    safe(stage, "source.json");
    const Json identity{{"endpoint", remote(repo.source.provider).url}, {"repo", repo.source.repo}, {"revision", repo.source.revision}, {"destination", name}, {"files", entries}};
    if (fs::exists(manifest)) {
        if (read(manifest) != identity) throw Error(Error::conflict, "download_changed", "staged download belongs to a different revision or file selection");
    } else write_json(manifest, identity);
    std::uint64_t remaining = std::max(total, request.expected_bytes.value_or(total));
    for (std::size_t index = 0; index < repo.files.size(); ++index) {
        const auto& file = repo.files[index];
        const auto path = content / fs::u8path(file.name);
        safe(stage, path.lexically_relative(stage));
        const auto part = parts / std::to_string(index);
        auto partial = part; partial += ".partial";
        auto meta = partial; meta += ".json";
        safe(stage, part.lexically_relative(stage));
        safe(stage, partial.lexically_relative(stage)); safe(stage, meta.lexically_relative(stage));
        fs::create_directories(path.parent_path());
        if (fs::is_regular_file(path)) remaining -= std::min(file.size, fs::file_size(path));
        else if (fs::is_regular_file(part)) remaining -= std::min(file.size, fs::file_size(part));
        else if (fs::is_regular_file(partial)) remaining -= std::min(file.size, fs::file_size(partial));
    }
    if (fs::space(root).available < remaining) throw Error(Error::io, "download_disk_full", "not enough free space for the selected files");
    std::mutex progress_mutex;
    std::vector<std::uint64_t> downloaded(repo.files.size());
    std::size_t completed = 0;
    auto reported = std::chrono::steady_clock::time_point{};
    auto report = [&](std::size_t index, std::uint64_t bytes, bool done) {
        std::lock_guard lock(progress_mutex);
        downloaded[index] = bytes; completed += done;
        const auto now = std::chrono::steady_clock::now();
        if (control.progress && (done || now - reported >= std::chrono::milliseconds(100))) {
            std::uint64_t sum = 0; for (auto value : downloaded) sum += value;
            control.progress(sum, total, completed); reported = now;
        }
    };
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false}; std::exception_ptr failure;
    Control active{[&] { return failed.load() || (control.cancelled && control.cancelled()); }, {}};
    auto work = [&] {
        try {
            for (;;) {
                active.checkpoint(); const auto index = next.fetch_add(1); if (index >= repo.files.size()) break;
                const auto& file = repo.files[index];
                const auto repo_name = hub::escape(repo.source.repo, true), revision = hub::escape(repo.source.revision);
                const auto url = repo.source.provider == Provider::hf ? "/" + repo_name + "/resolve/" + revision + "/" + hub::escape(file.name, true) :
                    "/api/v1/models/" + repo_name + "/repo?Revision=" + revision + "&FilePath=" + hub::escape(file.name);
                const auto ready = content / fs::u8path(file.name);
                hub::save({remote(repo.source.provider), network_.ca, request.direct}, url, file, fs::exists(ready) ? ready : parts / std::to_string(index), active,
                    [&](std::uint64_t bytes) { report(index, bytes, false); });
                report(index, file.size, true);
            }
        } catch (...) {
            std::lock_guard lock(progress_mutex);
            if (!failed.exchange(true)) failure = std::current_exception();
        }
    };
    std::vector<std::thread> workers;
    try { for (std::size_t i = 1; i < std::min(request.workers, repo.files.size()); ++i) workers.emplace_back(work); }
    catch (...) { failed = true; for (auto& worker : workers) worker.join(); throw; }
    work(); for (auto& worker : workers) worker.join();
    if (failure) std::rethrow_exception(failure);
    for (std::size_t index = 0; index < repo.files.size(); ++index) {
        control.checkpoint();
        const auto path = content / fs::u8path(repo.files[index].name);
        if (!fs::exists(path)) fs::rename(parts / std::to_string(index), path);
        else if (fs::exists(parts / std::to_string(index))) throw Error(Error::invalid, "invalid_download_path", "selected file names collide on this filesystem");
    }
    control.checkpoint(); safe(root, destination);
    if (fs::exists(root / destination)) throw Error(Error::conflict, "download_destination_exists", "download destination already exists");
    fs::create_directories((root / destination).parent_path());
    publish(content, root / destination);
    std::error_code ignored; fs::remove_all(stage, ignored);
    return {root / destination, repo.source, total, repo.files.size()};
} catch (const fs::filesystem_error&) {
    throw Error(Error::io, "download_io_failed", "download directory could not be read or written");
} catch (const Json::exception&) {
    throw Error(Error::integrity, "download_state_invalid", "saved download state is invalid; existing files were preserved");
}
}
