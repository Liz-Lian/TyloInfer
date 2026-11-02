#pragma once

#include "storage/storage.h"
#include "models/instances.h"
#include <nlohmann/json.hpp>

namespace mfq::server {

class Sessions {
public:
    using Json = nlohmann::json;
    struct Error : std::runtime_error {
        enum Kind { invalid, not_found, conflict, unavailable, too_large, stopping, save_failed } kind;
        std::string code;
        Json details;
        Error(Kind kind, std::string code, std::string message, Json details = Json::object())
            : std::runtime_error(std::move(message)), kind(kind), code(std::move(code)), details(std::move(details)) {}
    };
    class Reply {
    public:
        ~Reply();
        // End means execution and persistence completed; saving failures remain observable on every read.
        std::optional<std::vector<common_chat_msg_diff>> next(const std::function<bool()>& cancelled = {});
        bool cancel();
        Json finish();
        Json snapshot() const;
    private:
        friend class Sessions;
        struct State;
        explicit Reply(std::unique_ptr<State> state);
        std::unique_ptr<State> state_;
    };

    Sessions(Storage& storage, ModelInstances& models);
    ~Sessions();
    Json create(const Json& request);
    Json get(const std::string& id) const;
    Json list(std::size_t limit = 50, std::size_t offset = 0) const;
    Json update(const std::string& id, const Json& request);
    Json messages(const std::string& id) const;
    Json append(const std::string& id, const Json& request);
    Json fork(const std::string& id, const Json& request);
    Json rewind(const std::string& id, const Json& request);
    void erase(const std::string& id);
    Json export_archive(const std::string& id) const;
    Json import_archive(const Json& archive);
    Json presets() const;
    Json put_preset(const Json& request, const std::string& id = {});
    void erase_preset(const std::string& id);
    std::shared_ptr<Reply> start(const std::string& id, Json request);
    Json responses(const std::string& id, std::size_t limit = 200) const;
    Json cancel(const std::string& id);
    void request_stop();
    void shutdown();
private:
    Storage& storage_;
    ModelInstances& models_;
    std::mutex mutex_;
    std::map<std::string, std::weak_ptr<Reply>> active_;
    bool stopping_ = false;
    void release(const std::string& id, const Reply::State* state);
    void save(Reply::State& state);
};

} // namespace mfq::server
