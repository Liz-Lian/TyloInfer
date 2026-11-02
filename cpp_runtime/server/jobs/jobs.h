#pragma once

#include "storage/storage.h"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <nlohmann/json.hpp>

namespace mfq::server {
class Jobs {
public:
    using Json = nlohmann::json;
    struct Error : std::runtime_error {
        enum Kind { not_found, full, expired, invalid, conflict, stopping } kind;
        Error(Kind kind, const char* message) : std::runtime_error(message), kind(kind) {}
    };
    struct Cancelled {};
    struct Control {
        const std::atomic<bool>& cancelled;
        std::function<void(const std::string&, double, Json)> report;
        void progress(const std::string& phase, double value = 0, Json data = Json::object()) const { report(phase, value, std::move(data)); }
        void checkpoint() const { if (cancelled) throw Cancelled{}; }
    };
    using Work = std::function<Json(const Control&)>;
    using Failure = std::function<Json(std::exception_ptr)>;
    explicit Jobs(Storage& storage, Failure failure);
    ~Jobs();
    std::string submit(std::string kind, Json payload, Work work);
    Json get(const std::string& id) const;
    Json list(std::size_t limit = 100, std::size_t offset = 0, const std::string& status = {}, const std::string& kind = {}) const;
    void erase(const std::string& id);
    std::size_t clear();
    Json cancel(const std::string& id);
    Json events(const std::string& id, std::int64_t after, std::size_t limit = 256) const;
    void request_stop();
    void shutdown();
    static bool terminal(const Json& job);
private:
    struct Task {
        std::string id;
        Work work;
        std::shared_ptr<std::atomic<bool>> cancelled = std::make_shared<std::atomic<bool>>(false);
    };
    void update(const std::string& id, const char* state, Json result = nullptr, Json error = nullptr);
    void progress(const std::string& id, const std::string& phase, double value, Json data);
    void loop() noexcept;
    Storage& storage_;
    Failure failure_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::shared_ptr<Task>> queued_;
    std::shared_ptr<Task> running_;
    bool stopping_ = false;
    std::exception_ptr worker_error_;
    std::thread worker_;
};
}
