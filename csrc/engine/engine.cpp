#include "engine/engine.h"
#include "engine/scheduler/scheduler.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <utility>

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/runtime_metrics.h"
#include "engine/models/prepare.h"
#include "engine/models/source.h"
#include "engine/scheduler/request.h"
#include "engine/budget.h"
#include "engine/generation/prefill_budget.h"
#include "engine/requests/processor.h"
#include "engine/weights/memory.h"
#include "engine/weights/cache.h"
#include "engine/state/session/sessions.h"
#include <stdexcept>
#include <type_traits>
#endif

namespace tyloi::engine {

Engine::Engine() = default;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
Engine::Engine(std::shared_ptr<models::CausalModel> model, std::unique_ptr<tyloi::Executor> executor,
        RuntimeConfig config, std::shared_ptr<const ModelSource> source,
        std::shared_ptr<weights::Memory> memory, std::unique_ptr<weights::Cache> weights,
        std::unique_ptr<models::State> model_state, std::unique_ptr<state::Sessions> sessions,
        std::unique_ptr<TextProcessor> text, models::ForwardOptions forward,
        std::unique_ptr<tyloi::Executor::Graph> graph) {
    rebind(std::move(model), std::move(executor), std::move(config), std::move(source),
        std::move(memory), std::move(weights), std::move(model_state), std::move(sessions),
        std::move(text), std::move(forward), std::move(graph));
    start();
}

// Wiring that needs this Engine: caches, budget, expert space, reservations.
void Engine::install(std::unique_ptr<tyloi::Executor::Graph> graph) {
    try {
        model_info_ = model_->info();
        info_ = {};
        auto& info = info_;
        info.max_requests = std::max<std::size_t>(1, runtime_.continuous_batch.max_sequences);
        info.max_context = model_info_.max_context;
        info.vocab_size = static_cast<std::int32_t>(model_info_.vocab_size);
        info.model_type = model_info_.model_type;
        info.stop_token_ids = runtime_.stop_token_ids;
        info.capabilities = model_->capabilities();
        const auto& caps = info.capabilities;
        info.duplex = caps.full_duplex;
        info.multimodal = caps.image_input || caps.video_input || caps.audio_input || caps.audio_output || caps.full_duplex;
        info.reload = true;
        if (text_) info.chat = text_->chat_template_capabilities();
        auto scope = executor_->enter();
        tyloi::weights::LoadScope memory_load(*ledger_);
        graph_.emplace(model_info_.max_context, Device(kCUDA, forward_options_.primary), std::move(graph));
        budget_.emplace(*ledger_, *weights_, *cache_, Device(kCUDA, forward_options_.primary));
        budget_->bind();
        if (const auto bytes = runtime_.resident_budget_bytes) {
            engine::SessionCommand command; command.kind = engine::SessionCommand::Kind::memory_budget; command.bytes = bytes;
            set_budget(command);
        }
        budget_->prepare(Budget::Stage::loaded, source_.get());
        ready_ = true;
    } catch (...) {
        const auto failure = std::current_exception();
        if (graph) {
            try {
                auto scope = executor_->enter();
                graph_.emplace(0, Device(kCUDA, forward_options_.primary), std::move(graph));
            } catch (...) { graph.reset(); }
        }
        try { unload_model(); }
        catch (const std::exception& error) { std::cerr << "runtime_load_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "runtime_load_cleanup error=unknown\n"; }
        std::rethrow_exception(failure);
    }
}

void Engine::rebind(std::shared_ptr<models::CausalModel> model, std::unique_ptr<tyloi::Executor> executor,
        RuntimeConfig config, std::shared_ptr<const ModelSource> source,
        std::shared_ptr<weights::Memory> memory, std::unique_ptr<weights::Cache> weights,
        std::unique_ptr<models::State> model_state, std::unique_ptr<state::Sessions> sessions,
        std::unique_ptr<TextProcessor> text, models::ForwardOptions forward,
        std::unique_ptr<tyloi::Executor::Graph> graph) {
    std::exception_ptr failure;
    if (executor_) {
        try { unload_model(); }
        catch (...) { failure = std::current_exception(); }
    }
    model_ = std::move(model);
    executor_ = std::move(executor);
    runtime_ = std::move(config);
    source_ = std::move(source);
    ledger_ = std::move(memory);
    weights_ = std::move(weights);
    idle_state_ = std::move(model_state);
    cache_ = std::move(sessions);
    text_ = std::move(text);
    forward_options_ = std::move(forward);
    if (failure) {
        try {
            auto scope = executor_->enter();
            graph_.emplace(0, Device(kCUDA, forward_options_.primary), std::move(graph));
        } catch (...) { graph.reset(); }
        try { unload_model(); }
        catch (const std::exception& error) { std::cerr << "runtime_load_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "runtime_load_cleanup error=unknown\n"; }
        std::rethrow_exception(failure);
    }
    install(std::move(graph));
}
#endif
Engine::~Engine() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    if (scheduler_ || executor_) {
        try { shutdown(); }
        catch (const std::exception& error) { std::cerr << "runtime_unload_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "runtime_unload_cleanup error=unknown\n"; }
    }
#endif
}

#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
void Engine::retire_request(Request& request) noexcept {
    request.generation = {};
    request.state.reset();
}
#else
EngineInfo Engine::describe() const { return info_; }

bool Engine::ready() const { return ready_; }

std::unique_ptr<tyloi::Executor::Scope> Engine::enter() { return executor_->enter(); }

void Engine::prepare_request(Request& request, bool batched) {
    engine::prepare_request(request, text_.get(), info_, batched, false);
    if (!idle_state_) {
        request.output.reset();
        request.input = {};
        throw std::logic_error("native engine state is already in use");
    }
    request.state = std::move(idle_state_);
}

void Engine::retire_request(Request& request) noexcept {
    request.generation = {};
    if (model_ && request.state) idle_state_ = std::move(request.state);
    else request.state.reset();
}

SessionResult Engine::session_command(const SessionCommand& command) {
    if (!ready_) throw std::runtime_error("CUDA engine is unloaded");
    auto scope = executor_->enter();
    return command.kind == engine::SessionCommand::Kind::memory_budget
        ? set_budget(command) : cache_->control(command);
}

void Engine::drain() {
    auto scope = executor_->enter();
    std::exception_ptr failure;
    try { weights_->drain(); }
    catch (...) { failure = std::current_exception(); }
    try { executor_->drain(); }
    catch (...) { if (!failure) failure = std::current_exception(); }
    if (failure) std::rethrow_exception(failure);
}

void Engine::unload_model() {
    ready_ = false;
    std::exception_ptr failure;
    const auto remember = [&](std::exception_ptr error) { if (!failure) failure = std::move(error); };
    text_.reset();
    if (executor_) {
        try { drain(); }
        catch (...) { remember(std::current_exception()); }
        std::unique_ptr<tyloi::Executor::Scope> scope;
        try { scope = executor_->enter(); }
        catch (...) { remember(std::current_exception()); }
        clear_bindings();
        graph_.reset();
        try { executor_->clear(); }
        catch (...) { remember(std::current_exception()); }
        idle_state_.reset(); model_.reset();
        weights_->clear_workspace();
        budget_.reset(); cache_.reset();
        weights_.reset();
        scope.reset();
        executor_.reset();
    }
    ledger_.reset(); source_.reset();
    if (failure) std::rethrow_exception(failure);
}

ControlResult Engine::control_command(ControlRequest request) {
    if (!ready_) throw std::runtime_error("engine is unloaded");
    auto scope = executor_->enter();
    return std::visit([&](auto&& value) -> engine::ControlResult {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, engine::DecodeTokens> || std::is_same_v<T, engine::CountTokens> ||
                      std::is_same_v<T, engine::PrepareDuplex> || std::is_same_v<T, engine::PrepareDuplexStep>) {
            if (!text_) throw std::invalid_argument("text control requires a tokenizer");
            if constexpr (std::is_same_v<T, engine::DecodeTokens>) return text_->decode_tokens(value.tokens, value.excluded);
            else if constexpr (std::is_same_v<T, engine::CountTokens>) return engine::TokenCount{text_->count_tokens(value.prompt), info_.max_context};
            else if constexpr (std::is_same_v<T, engine::PrepareDuplex>) {
                text_->prepare_duplex_session(value.prompt, value.parameters); return std::move(value.parameters);
            } else { text_->prepare_duplex_step(value.text, value.input); return std::move(value.input); }
        } else if constexpr (std::is_same_v<T, engine::RuntimeMetrics>) {
            return runtime_metrics(model_->metrics(), {}, std::nullopt, false, 3,
                {}, {}, std::nullopt, {}, source_.get(),
                *ledger_, *weights_, *cache_, *budget_,
                executor_->memory(runtime_.devices), Device(kCUDA, forward_options_.primary), runtime_.online_budget);
        } else throw std::invalid_argument("model has no duplex component");
    }, std::move(request));
}

bool Engine::can_batch(const EngineRequest&) const { return false; }

bool Engine::exclusive() const { return false; }

void Engine::execute(const std::vector<RequestId>&) {}

void Engine::clear_bindings() noexcept {
    if (budget_) budget_->unbind();
}

void Engine::reserve(models::State& state, std::int64_t tokens) {
    std::size_t bytes = 0;
    const Device primary(kCUDA, forward_options_.primary);
    for (const auto& [device, growth] : state.growth(tokens, forward_options_))
        if (device == primary) bytes = prefill_budget_add(bytes, growth);
    if (bytes)
        budget_->reserve_kv(prefill_budget_add(bytes, std::size_t(64) << 20));
}

SessionResult Engine::set_budget(const SessionCommand& command) {
    if (!runtime_.online_budget)
        throw std::invalid_argument("online memory budgeting requires a single-device CUDA worker");
    auto change = budget_->change(command);
    weights_->settle();
    graph_->invalidate();
    if (ready_) reset_states(1);
    if (idle_state_) idle_state_->reset(1);
    return change.apply();
}

#endif

} // namespace tyloi::engine
