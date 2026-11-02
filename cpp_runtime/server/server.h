#pragma once

#include "models/factory.h"

#include <atomic>
#include <filesystem>
#include <string>
#include <vector>

namespace mfq::server {

struct Config {
    std::string host = "127.0.0.1";
    int port = 8090;
    std::filesystem::path data_dir = ".tyloinfer";
    std::filesystem::path web_root;
    std::filesystem::path database;
    std::vector<std::filesystem::path> model_dirs;
    std::string api_key_env = "TYLOINFER_SERVER_API_KEY";
    std::string instance_id;
    bool exit_on_stdin_close = false;
};

int run(const Config& config, const std::atomic<bool>& stop_requested, EngineFactory factory = {});

} // namespace mfq::server
