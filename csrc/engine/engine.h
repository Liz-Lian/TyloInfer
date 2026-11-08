#pragma once

#include "models/metadata.h"
#include "requests/request.h"
#include "requests/result.h"
#include "executor.h"
#include "generation/sequence.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "config.h"
#include "budget.h"
#include "generation/decode.h"
#include "models/causal_model.h"
#endif

#include <chrono>
#include <future>
#include <memory>
#include <optional>
#include <utility>
#include <variant>

namespace tyloi::models { class CausalModel; class State; }
namespace tyloi { class ModelSource; }
namespace tyloi::weights { class Memory; class Cache; }
namespace tyloi::state { class Sessions; }

namespace tyloi::engine {

class Scheduler;
class InferenceOutput;
class TextProcessor;
struct Request;

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
#ifdef TYLOINFER_NATIVE_CUDA
    Engine(std::shared_ptr<models::CausalModel> model, std::unique_ptr<tyloi::Executor> executor,
        Config config, std::shared_ptr<const ModelSource> source,
        std::shared_ptr<weights::Memory> memory, std::unique_ptr<weights::Cache> weights,
        std::unique_ptr<models::State> model_state, std::unique_ptr<state::Sessions> sessions,
        std::unique_ptr<TextProcessor> text, models::ForwardOptions forward,
        std::unique_ptr<tyloi::Executor::Graph> graph);
    // Retire active requests before replacing dependencies; the Scheduler remains attached.
    void rebind(std::shared_ptr<models::CausalModel> model, std::unique_ptr<tyloi::Executor> executor,
        Config config, std::shared_ptr<const ModelSource> source,
        std::shared_ptr<weights::Memory> memory, std::unique_ptr<weights::Cache> weights,
        std::unique_ptr<models::State> model_state, std::unique_ptr<state::Sessions> sessions,
        std::unique_ptr<TextProcessor> text, models::ForwardOptions forward,
        std::unique_ptr<tyloi::Executor::Graph> graph);
#endif
    // Start one consumer before enable_reload(); next_reload() delivers posted work before shutdown.
    // Complete reply only after binding and its execution scopes have exited.
    struct Reload { std::int64_t context; std::promise<void> reply; };
    struct Snapshot {
        EngineStatus status;
        Metrics engine, session;
        Clock::time_point captured;
    };
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
    Engine();
    // Start after loading; shut down before derived members are destroyed.
    void start();
#ifdef TYLOINFER_NATIVE_CUDA
    virtual EngineInfo describe() const;
    virtual bool ready() const;
    virtual bool can_batch(const EngineRequest& request) const;
    virtual bool exclusive() const;
    virtual std::unique_ptr<tyloi::Executor::Scope> enter();
    virtual void prepare_request(Request& request, bool batched);
    virtual tyloi::StepSequence<EventData> generate(const RequestId& id, Request& request);
#else
    virtual EngineInfo describe() const = 0;
    virtual bool ready() const = 0;
    virtual bool can_batch(const EngineRequest& request) const = 0;
    virtual bool exclusive() const = 0;
    virtual std::unique_ptr<tyloi::Executor::Scope> enter() = 0;
    virtual void prepare_request(Request& request, bool batched) = 0;
    virtual tyloi::StepSequence<EventData> generate(const RequestId& id, Request& request) = 0;
#endif
    virtual void retire_request(Request& request) noexcept;
    void reset_states(std::int64_t batch);
#ifdef TYLOINFER_NATIVE_CUDA
    virtual void execute(const std::vector<RequestId>& eligible);
    virtual SessionResult session_command(const SessionCommand& command);
    virtual void unload_model();
    virtual ControlResult control_command(ControlRequest request);
#else
    virtual void execute(const std::vector<RequestId>& eligible) = 0;
    virtual SessionResult session_command(const SessionCommand& command) = 0;
    virtual void unload_model() = 0;
    virtual ControlResult control_command(ControlRequest request) = 0;
#endif

private:
    friend class Scheduler;
#ifdef TYLOINFER_NATIVE_CUDA
    void install(std::unique_ptr<tyloi::Executor::Graph> graph);

    SessionResult set_budget(const SessionCommand&);
    void drain();
    void clear_bindings() noexcept;
    std::shared_ptr<models::CausalModel> model_;
    std::unique_ptr<tyloi::Executor> executor_;
    Config config_;
    std::shared_ptr<const ModelSource> source_;
    std::shared_ptr<weights::Memory> ledger_;
    std::unique_ptr<weights::Cache> weights_;
    std::unique_ptr<models::State> idle_state_;
    models::ForwardOptions forward_options_{};
    models::Info model_info_{};
    std::optional<Decode> graph_;
    std::unique_ptr<state::Sessions> cache_;
    std::optional<Budget> budget_;
    std::unique_ptr<TextProcessor> text_;
    EngineInfo info_;
    bool ready_ = false;
#endif
    std::unique_ptr<Scheduler> scheduler_;
};

} // namespace tyloi::engine
