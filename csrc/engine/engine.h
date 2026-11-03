#pragma once

#include "models/capabilities.h"
#include "requests/request.h"
#include "requests/result.h"
#include "executor.h"
#include "generation/sequence.h"

#include <chrono>
#include <future>
#include <memory>
#include <optional>
#include <variant>

namespace mfq::engine {

class Scheduler;
struct ExecutionRequest;

struct EngineInfo {
    std::size_t max_requests = 1;
    std::int64_t max_context = 0;
    std::int32_t vocab_size = 0;
    bool multimodal = false, duplex = false, reload = false;
    ChatTemplateCapabilities chat;
    std::string model_type;
    MfqModelCapabilities capabilities;
    bool probability = false;
    std::unordered_set<std::int64_t> stop_token_ids;
};
struct EngineStatus {
    std::size_t available = 0;
    bool healthy = true;
};
class Engine {
public:
    // Start one consumer before enable_reload(); next_reload() delivers posted work before shutdown.
    // Complete reply only after binding and its execution scopes have exited.
    struct Reload { std::int64_t context; std::promise<void> reply; };
    struct Snapshot {
        EngineStatus status;
        Metrics runtime, session;
        Clock::time_point captured;
    };
    Engine();
    virtual ~Engine();
    std::shared_ptr<Output> submit(EngineRequest request) const;
    bool cancel(const std::string& id) const;
    bool cancel_session(const std::string& id) const;
    void cancel_all() const;
    EngineInfo info() const;
    EngineStatus status() const;
    Snapshot snapshot() const;
    ControlResult control(ControlRequest request) const;
    SessionResult session(SessionCommand command) const;
    std::int64_t reload(std::int64_t context) const;
    void enable_reload();
    std::optional<Reload> next_reload();
    void shutdown();

    bool supports_multimodal_generation() const { return info().multimodal; }
    bool supports_reload() const { return info().reload; }
    auto chat_template_capabilities() const { return info().chat; }
    std::int32_t vocab_size() const { return info().vocab_size; }
    void prepare_duplex_session(const std::string& prompt, MfqDuplexSessionParams& params) const;
    void prepare_duplex_step(const std::string& text, MfqDuplexStepInput& input) const;
    std::string decode_tokens(const std::vector<std::int64_t>& tokens,
        const std::unordered_set<std::int64_t>& excluded = {}) const;

protected:
    // Start after loading; shut down before derived members are destroyed.
    void start();
    virtual EngineInfo describe() const = 0;
    virtual bool ready() const = 0;
    virtual bool can_batch(const EngineRequest& request) const = 0;
    virtual bool exclusive() const = 0;
    virtual std::unique_ptr<mfq::Executor::Scope> enter() = 0;
    virtual std::unique_ptr<ExecutionRequest> prepare_request(EngineRequest request, bool batched) = 0;
    virtual mfq::StepSequence<EventData> generate(const RequestId& id, ExecutionRequest& request) = 0;
    virtual void execute(const std::vector<RequestId>& eligible) = 0;
    virtual SessionResult session_command(const SessionCommand& command) = 0;
    virtual void unload_model() = 0;
    virtual ControlResult control_command(ControlRequest request) = 0;

private:
    friend class Scheduler;
    std::unique_ptr<Scheduler> scheduler_;
};

} // namespace mfq::engine
