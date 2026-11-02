#include "fetch.h"
#include "storage/files.h"
#include <curl/curl.h>
#include <openssl/evp.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <memory>
#include <sstream>

namespace mfq::server::hub {
namespace {
using Error = Hub::Error;
using Json = nlohmann::json;
namespace fs = std::filesystem;
struct Global {
    Global() { if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) throw Error(Error::network, "network_unavailable", "could not initialize libcurl"); }
    ~Global() { curl_global_cleanup(); }
};
template<class Value> void set(CURL* handle, CURLoption option, Value value) {
    if (curl_easy_setopt(handle, option, value) != CURLE_OK)
        throw Error(Error::network, "network_unavailable", "could not configure libcurl");
}
using Handle = std::unique_ptr<CURL, decltype(&curl_easy_cleanup)>;
Handle open(const Fetch& fetch, const std::string& path) {
    static Global global;
    Handle handle(curl_easy_init(), curl_easy_cleanup);
    if (!handle) throw Error(Error::network, "network_unavailable", "could not create an HTTP request");
    const auto url = fetch.remote.url + path;
    set(handle.get(), CURLOPT_URL, url.c_str());
    set(handle.get(), CURLOPT_PROTOCOLS_STR, "http,https");
    set(handle.get(), CURLOPT_REDIR_PROTOCOLS_STR, url.rfind("https://", 0) == 0 ? "https" : "http,https");
    set(handle.get(), CURLOPT_FOLLOWLOCATION, 1L);
    set(handle.get(), CURLOPT_MAXREDIRS, 5L);
    set(handle.get(), CURLOPT_SUPPRESS_CONNECT_HEADERS, 1L);
    set(handle.get(), CURLOPT_SSL_VERIFYPEER, 1L);
    set(handle.get(), CURLOPT_SSL_VERIFYHOST, 2L);
    set(handle.get(), CURLOPT_NOSIGNAL, 1L);
    set(handle.get(), CURLOPT_CONNECTTIMEOUT_MS, 10000L);
    set(handle.get(), CURLOPT_LOW_SPEED_LIMIT, 1L);
    set(handle.get(), CURLOPT_LOW_SPEED_TIME, 30L);
    set(handle.get(), CURLOPT_USERAGENT, "TyloInfer-native");
    set(handle.get(), CURLOPT_ACCEPT_ENCODING, "identity");
    if (!fetch.ca.empty()) set(handle.get(), CURLOPT_CAINFO, fetch.ca.string().c_str());
    if (fetch.direct) set(handle.get(), CURLOPT_PROXY, "");
    if (!fetch.remote.token.empty()) {
        set(handle.get(), CURLOPT_HTTPAUTH, CURLAUTH_BEARER);
        set(handle.get(), CURLOPT_XOAUTH2_BEARER, fetch.remote.token.c_str());
    }
    return handle;
}
struct Transfer {
    const Hub::Control& control;
    const Hub::File* file = nullptr;
    fs::path meta;
    std::string identity, text, etag, saved_etag, range;
    std::ofstream output;
    std::function<void(std::uint64_t)> report;
    std::uint64_t offset = 0, written = 0;
    long status = 0;
    bool restart = false, prepared = false;
    std::exception_ptr failure;
    explicit Transfer(const Hub::Control& control) : control(control) {}
    template<class Action> std::size_t call(std::size_t size, Action action) noexcept {
        try { control.checkpoint(); action(); return size; }
        catch (...) { if (!failure) failure = std::current_exception(); return 0; }
    }
};
std::string trim(std::string value) {
    const auto begin = value.find_first_not_of(" \t\r\n"), end = value.find_last_not_of(" \t\r\n");
    return begin == std::string::npos ? std::string() : value.substr(begin, end - begin + 1);
}
std::size_t header(char* bytes, std::size_t size, std::size_t count, void* data) noexcept {
    auto& transfer = *static_cast<Transfer*>(data);
    return transfer.call(size * count, [&] {
        std::string line(bytes, size * count);
        if (line.rfind("HTTP/", 0) == 0) {
            std::istringstream status(line); std::string protocol; status >> protocol >> transfer.status;
            transfer.etag.clear(); transfer.range.clear(); transfer.prepared = false;
        } else if (const auto colon = line.find(':'); colon != std::string::npos) {
            auto name = line.substr(0, colon);
            std::transform(name.begin(), name.end(), name.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
            if (name == "etag") {
                transfer.etag = trim(line.substr(colon + 1));
                if (transfer.etag.size() > 1024) throw Error(Error::integrity, "invalid_download_response", "ETag exceeds its limit");
                if (transfer.etag.rfind("W/", 0) == 0) transfer.etag.clear();
            } else if (name == "content-range") transfer.range = trim(line.substr(colon + 1));
        }
        if (!transfer.file || line != "\r\n" || (transfer.status != 200 && transfer.status != 206)) return;
        if (transfer.offset && transfer.status == 200) { transfer.restart = true; throw Error(Error::integrity, "range_ignored", "server ignored the resume range"); }
        if (transfer.status == 206) {
            std::uint64_t begin = 0, end = 0, total = 0; char dash = 0, slash = 0; std::string unit;
            std::istringstream range(transfer.range); range >> unit >> begin >> dash >> end >> slash >> total;
            if (!range || range.peek() != std::char_traits<char>::eof() || unit != "bytes" || dash != '-' || slash != '/' || begin != transfer.offset ||
                end < begin || end >= total || total != transfer.file->size)
                throw Error(Error::integrity, "invalid_download_range", "download returned an inconsistent byte range");
            if (!transfer.saved_etag.empty() && transfer.etag != transfer.saved_etag)
                throw Error(Error::integrity, "download_changed", "download identity changed while resuming");
        }
        write_json(transfer.meta, {{"identity", transfer.identity}, {"etag", transfer.etag}});
        transfer.prepared = true;
    });
}
std::size_t body(char* bytes, std::size_t size, std::size_t count, void* data) noexcept {
    auto& transfer = *static_cast<Transfer*>(data);
    return transfer.call(size * count, [&] {
        if (transfer.status >= 300 && transfer.status < 400) return;
        if (transfer.status != 200 && transfer.status != 206) return;
        const auto length = size * count;
        if (!transfer.file) {
            if (length > 16 * 1024 * 1024 - transfer.text.size())
                throw Error(Error::integrity, "hub_response_too_large", "repository metadata exceeds 16 MiB");
            transfer.text.append(bytes, length); return;
        }
        if (!transfer.prepared || length > transfer.file->size - transfer.offset - transfer.written)
            throw Error(Error::integrity, "download_size_mismatch", "download exceeds the declared file size");
        transfer.output.write(bytes, static_cast<std::streamsize>(length));
        if (!transfer.output) throw Error(Error::io, "download_write_failed", "could not write downloaded bytes");
        transfer.written += length;
        if (transfer.report) transfer.report(transfer.offset + transfer.written);
    });
}
int progress(void* data, curl_off_t, curl_off_t, curl_off_t, curl_off_t) noexcept {
    auto& transfer = *static_cast<Transfer*>(data);
    try { transfer.control.checkpoint(); return 0; }
    catch (...) { if (!transfer.failure) transfer.failure = std::current_exception(); return 1; }
}
void perform(CURL* handle, Transfer& transfer) {
    set(handle, CURLOPT_HEADERFUNCTION, header); set(handle, CURLOPT_HEADERDATA, &transfer);
    set(handle, CURLOPT_WRITEFUNCTION, body); set(handle, CURLOPT_WRITEDATA, &transfer);
    set(handle, CURLOPT_NOPROGRESS, 0L);
    set(handle, CURLOPT_XFERINFOFUNCTION, progress); set(handle, CURLOPT_XFERINFODATA, &transfer);
    const auto result = curl_easy_perform(handle);
    if (transfer.failure) std::rethrow_exception(transfer.failure);
    if (result != CURLE_OK) throw Error(Error::network, "download_connection_failed", curl_easy_strerror(result));
    if (transfer.status != 200 && transfer.status != 206)
        throw Error(transfer.status == 404 ? Error::missing : Error::network, "hub_http_error", "repository returned HTTP " + std::to_string(transfer.status));
}
}
std::string escape(const std::string& value, bool slashes) {
    constexpr char digits[] = "0123456789ABCDEF";
    std::string result;
    for (unsigned char ch : value) {
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') ||
            ch == '-' || ch == '_' || ch == '.' || ch == '~' || (slashes && ch == '/')) result += static_cast<char>(ch);
        else { result += '%'; result += digits[ch >> 4]; result += digits[ch & 15]; }
    }
    return result;
}
std::string get(const Fetch& fetch, const std::string& path, const Hub::Control& control) {
    auto handle = open(fetch, path); set(handle.get(), CURLOPT_TIMEOUT_MS, 30000L);
    Transfer transfer(control); perform(handle.get(), transfer); return std::move(transfer.text);
}
std::string hash(const fs::path& path, const Hub::Control& control) {
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> digest(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!digest || EVP_DigestInit_ex(digest.get(), EVP_sha256(), nullptr) != 1)
        throw Error(Error::io, "checksum_failed", "could not initialize file checksum");
    std::ifstream input(path, std::ios::binary);
    if (!input) throw Error(Error::io, "download_read_failed", "could not read downloaded file");
    std::array<char, 64 * 1024> buffer;
    while (input) {
        control.checkpoint();
        input.read(buffer.data(), buffer.size());
        if (input.gcount() && EVP_DigestUpdate(digest.get(), buffer.data(), static_cast<std::size_t>(input.gcount())) != 1)
            throw Error(Error::io, "checksum_failed", "could not checksum downloaded file");
    }
    if (!input.eof()) throw Error(Error::io, "download_read_failed", "could not read downloaded file");
    std::array<unsigned char, EVP_MAX_MD_SIZE> bytes; unsigned length = 0;
    if (EVP_DigestFinal_ex(digest.get(), bytes.data(), &length) != 1 || length != 32)
        throw Error(Error::io, "checksum_failed", "could not finalize file checksum");
    constexpr char digits[] = "0123456789abcdef"; std::string result;
    for (unsigned i = 0; i < length; ++i) { result += digits[bytes[i] >> 4]; result += digits[bytes[i] & 15]; }
    return result;
}
void save(const Fetch& fetch, const std::string& path, const Hub::File& file,
        const fs::path& destination, const Hub::Control& control, const std::function<void(std::uint64_t)>& report) {
    auto partial = destination; partial += ".partial";
    auto meta = partial; meta += ".json";
    const auto identity = Json{{"url", fetch.remote.url + path}, {"size", file.size}, {"sha256", file.sha256}}.dump();
    for (int attempt = 0; attempt < 2; ++attempt) {
        control.checkpoint();
        Transfer transfer(control); transfer.file = &file; transfer.meta = meta; transfer.identity = identity; transfer.report = report;
        if (fs::exists(destination)) {
            if (fs::file_size(destination) == file.size && (file.sha256.empty() || hash(destination, control) == file.sha256)) {
                if (report) report(file.size); return;
            }
            throw Error(Error::integrity, "download_changed", "a staged file no longer matches its metadata");
        }
        if (fs::exists(partial) && fs::exists(meta)) {
            if (fs::file_size(meta) > 16384) throw Error(Error::integrity, "download_state_invalid", "partial file metadata is too large");
            std::ifstream saved(meta); const auto value = Json::parse(saved);
            if (value.value("identity", "") != identity) throw Error(Error::conflict, "download_changed", "partial file belongs to a different source");
            transfer.offset = fs::file_size(partial);
            transfer.saved_etag = value.value("etag", "");
            if (std::any_of(transfer.saved_etag.begin(), transfer.saved_etag.end(), [](unsigned char ch) { return ch < 32 || ch == 127; }))
                throw Error(Error::integrity, "download_state_invalid", "partial file has an invalid ETag");
            if (transfer.offset > file.size) throw Error(Error::integrity, "download_size_mismatch", "partial file exceeds its declared size");
            if (transfer.offset && transfer.saved_etag.empty() && file.sha256.empty()) transfer.offset = 0;
        }
        if (transfer.offset != file.size || !fs::exists(partial)) {
            auto handle = open(fetch, path);
            transfer.output.open(partial, std::ios::binary | (transfer.offset ? std::ios::app : std::ios::trunc));
            if (!transfer.output) throw Error(Error::io, "download_write_failed", "could not open partial download");
            std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> headers(nullptr, curl_slist_free_all);
            if (transfer.offset) {
                set(handle.get(), CURLOPT_RESUME_FROM_LARGE, static_cast<curl_off_t>(transfer.offset));
                if (!transfer.saved_etag.empty()) {
                    headers.reset(curl_slist_append(nullptr, ("If-Range: " + transfer.saved_etag).c_str()));
                    if (!headers) throw std::bad_alloc();
                    set(handle.get(), CURLOPT_HTTPHEADER, headers.get());
                }
            }
            try { perform(handle.get(), transfer); }
            catch (...) {
                transfer.output.close();
                if (transfer.restart && attempt == 0) { fs::remove(partial); fs::remove(meta); continue; }
                throw;
            }
            transfer.output.close();
            if (!transfer.output) throw Error(Error::io, "download_write_failed", "could not finish writing downloaded file");
        }
        if (fs::file_size(partial) != file.size) throw Error(Error::integrity, "download_size_mismatch", "downloaded file has the wrong size");
        if (!file.sha256.empty() && hash(partial, control) != file.sha256) {
            fs::remove(partial); fs::remove(meta);
            throw Error(Error::integrity, "download_checksum_mismatch", "downloaded file failed SHA-256 validation");
        }
        control.checkpoint(); fs::rename(partial, destination); fs::remove(meta);
        if (report) report(file.size); return;
    }
}
}
