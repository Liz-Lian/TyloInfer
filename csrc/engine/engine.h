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
#include <utility>
#include <variant>

namespace tyloi::models { class CausalModel; class State; }
namespace tyloi::engine { class Engine; struct EngineParts; }
namespace tyloi::cuda {
struct CudaEngineOptions;
std::unique_ptr<engine::Engine> load_cuda_engine(CudaEngineOptions);
void reload_cuda_engine(engine::Engine&, CudaEngineOptions);
// Assembles everything an Engine needs without touching the Engine itself.
namespace internal { engine::EngineParts build_engine_parts(CudaEngineOptions); }
}

namespace tyloi::engine {

class Scheduler;
class InferenceOutput;
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
    // Start one consumer before enable_reload(); next_reload() delivers posted work before shutdown.
    // Complete reply only after binding and its execution scopes have exited.
    struct Reload { std::int64_t context; std::promise<void> reply; };
    struct Snapshot {
        EngineStatus status;
        Metrics runtime, session;
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
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
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
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
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
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    friend struct EngineParts;
    // Builds the parts only; it never touches an Engine.
    friend EngineParts tyloi::cuda::internal::build_engine_parts(tyloi::cuda::CudaEngineOptions);
    friend std::unique_ptr<Engine> tyloi::cuda::load_cuda_engine(tyloi::cuda::CudaEngineOptions);
    friend void tyloi::cuda::reload_cuda_engine(Engine&, tyloi::cuda::CudaEngineOptions);
    // The model and executor enter through the constructor; nothing writes
    // Engine fields from outside.
    explicit Engine(EngineParts&& parts);
    // Device-side wiring that needs this Engine: caches, budget, reservations.
    void install();
    void rebuild(EngineParts&& parts);

    SessionResult set_budget(const SessionCommand&);
    void reserve(models::State& state, std::int64_t tokens);
    void drain();
    void clear_bindings() noexcept;
    struct Impl;
    std::shared_ptr<models::CausalModel> model_;
    std::unique_ptr<tyloi::Executor> executor_;
    std::unique_ptr<Impl> impl_;
#endif
    std::unique_ptr<Scheduler> scheduler_;
};

} // namespace tyloi::engine
