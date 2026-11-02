#pragma once

#include "hub.h"

namespace mfq::server::hub {
struct Fetch {
    Hub::Remote remote;
    std::filesystem::path ca;
    bool direct = false;
};
std::string get(const Fetch& fetch, const std::string& path, const Hub::Control& control);
void save(const Fetch& fetch, const std::string& path, const Hub::File& file,
    const std::filesystem::path& destination, const Hub::Control& control,
    const std::function<void(std::uint64_t)>& progress);
std::string escape(const std::string& value, bool slashes = false);
std::string hash(const std::filesystem::path& path, const Hub::Control& control);
}
