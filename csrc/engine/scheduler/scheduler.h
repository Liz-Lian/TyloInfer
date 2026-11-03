#pragma once

#include "engine/engine.h"
#include "request.h"

#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>

namespace mfq::engine {

class Scheduler {
public:
    struct Limits { std::size_t events = 64, bytes = 1024 * 1024; };
    explicit Scheduler(Engine& engine);
    Scheduler(Engine& engine, Limits limits);
    ~Scheduler();
    std::shared_ptr<Output> submit(EngineRequest request) const;
    bool cancel_request(const std::string& id) const;
    bool cancel_session(const std::string& id) const;
    void cancel_all() const;
    void shutdown();
    EngineInfo info() const;
    EngineStatus status() const;
    Engine::Snapshot snapshot() const;
    ControlResult control(ControlRequest request) const;
    SessionResult session(SessionCommand command) const;
    std::int64_t reload(std::int64_t context) const;
    void enable_reload();
    std::optional<Engine::Reload> next_reload();

private:
    friend class Engine;
    struct Submit { EngineRequest input; std::shared_ptr<Output> outbox; std::promise<void> reply; };
    struct Cancel { std::string id; bool session = false, all = false; std::promise<bool> reply; };
    struct Control { ControlRequest request; std::promise<ControlResult> reply; };
    struct Session { SessionCommand request; std::promise<SessionResult> reply; };
    struct Reload { std::int64_t context; std::promise<std::int64_t> reply; };
    struct Status { std::promise<EngineStatus> reply; };
    using Command = std::variant<Submit, Cancel, Control, Session, Reload, Status>;
    void enqueue(Command command) const;
    void loop() noexcept;
    void publish(Request& request, EngineEvent event);
    void flush(Request& request);
    void cancel(Request& request);
    bool has_active() const;
    EngineStatus execution_status() const;
    std::pair<bool, std::vector<EngineEvent>> advance(const std::vector<RequestId>& eligible);
    void retire_failed() noexcept;
    void reset_states(std::int64_t batch);
    void refresh_snapshot(bool healthy, bool blocked, bool metrics);

    Engine& engine_;
    Limits limits_;
    mutable std::mutex mutex_;
    std::shared_ptr<std::condition_variable> wake_ = std::make_shared<std::condition_variable>();
    std::condition_variable reload_wake_;
    std::optional<Engine::Reload> reload_job_;
    mutable std::deque<Command> mailbox_;
    EngineInfo info_;
    Engine::Snapshot snapshot_;
    mutable bool snapshot_requested_ = false;
    bool stopping_ = false;
    bool reload_enabled_ = false;
    bool healthy_ = true;
    std::thread worker_;
    std::unordered_map<std::string, Request> requests_;
    std::deque<std::string> order_;
};

} // namespace mfq::engine
