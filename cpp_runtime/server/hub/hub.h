#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mfq::server {
class Hub {
public:
    struct Error : std::runtime_error {
        enum Kind { invalid, missing, conflict, network, integrity, io, cancelled, unavailable } kind;
        const char* code;
        Error(Kind kind, const char* code, const std::string& message) : std::runtime_error(message), kind(kind), code(code) {}
    };
    enum class Provider { hf, ms };
    struct Source { Provider provider; std::string repo, revision; };
    struct File { std::string name, sha256; std::uint64_t size = 0; };
    struct Repo { Source source; std::string url; std::vector<File> files; };
    struct Remote { std::string url, token; };
    struct Network {
        Remote hf{"https://huggingface.co", {}}, ms{"https://modelscope.cn", {}};
        std::filesystem::path ca;
    };
    struct Control {
        std::function<bool()> cancelled;
        std::function<void(std::uint64_t, std::uint64_t, std::size_t)> progress;
        void checkpoint() const;
    };
    struct Request {
        Source source;
        std::string destination;
        std::optional<std::filesystem::path> directory;
        std::vector<std::string> include, exclude;
        std::size_t workers = 8;
        bool direct = false;
        std::optional<std::uint64_t> expected_bytes;
    };
    struct Result { std::filesystem::path path; Source source; std::uint64_t bytes; std::size_t files; };
    struct Directory { std::filesystem::path selected, actual, fallback; std::optional<std::filesystem::space_info> space; };
    Hub(std::filesystem::path workspace, Network network);
    Repo info(Source source, const Control& control = {}, bool direct = false) const;
    Result download(const Request& request, const Control& control = {});
    Directory directory() const;
    Directory configure(const std::optional<std::filesystem::path>& path);
private:
    std::filesystem::path workspace_, fallback_, selected_;
    std::vector<std::filesystem::path> roots_;
    Network network_;
    mutable std::mutex mutex_;
    std::mutex transfer_;
    Remote remote(Provider provider) const;
};
}
