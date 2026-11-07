#include "engine/engine.h"
#include "engine/scheduler/scheduler.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <utility>

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/impl.h"
#include "engine/runtime_metrics.h"
#include "engine/models/prepare.h"
#include "engine/scheduler/request.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/budget.h"
#include "engine/generation/prefill_budget.h"
#include "engine/models/common/session/cache.h"
#include "engine/requests/processor.h"
#include "engine/weights/memory.h"
#include <stdexcept>
#include <type_traits>
#endif

namespace tyloi::engine {

Engine::Engine() = default;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
Engine::Engine(EngineParts&& parts)
    : model_(std::move(parts.model)), executor_(std::move(parts.executor)), impl_(std::move(parts.impl)) {}

// Wiring that needs this Engine: caches, budget, expert space, reservations.
void Engine::install() {
    auto& impl = *impl_;
    try {
        impl.model_info_ = model_->info();
        auto& info = impl.info_;
        info.max_requests = std::max<std::size_t>(1, impl.runtime_.continuous_batch.max_sequences);
        info.max_context = impl.model_info_.max_context;
        info.vocab_size = static_cast<std::int32_t>(impl.model_info_.vocab_size);
        info.model_type = impl.model_info_.model_type;
        info.capabilities = model_->capabilities();
        const auto& caps = info.capabilities;
        info.duplex = caps.full_duplex;
        info.multimodal = caps.image_input || caps.video_input || caps.audio_input || caps.audio_output || caps.full_duplex;
        info.reload = true;
        if (impl.text_) info.chat = impl.text_->chat_template_capabilities();
        auto scope = executor_->enter();
        tyloi::weights::LoadScope memory_load(*impl.ledger_);
        impl.graph_.emplace(impl.model_info_.max_context, Device(kCUDA, impl.forward_options_.primary), executor_->graph());
        impl.cache_ = cuda::internal::make_sessions(*impl.source_, impl.runtime_.session_cache,
            impl.runtime_.prefix_cache, impl.model_info_.max_context,
            impl.idle_state_->supports_text_session_state(),
            impl.idle_state_->supports_paged_text_session_state());
        impl.budget_.emplace(*impl.ledger_, *impl.weights_, *impl.cache_,
            Device(kCUDA, impl.forward_options_.primary));
        impl.budget_->bind();
        if (const auto bytes = cuda::runtime_options::resident_budget_bytes()) {
            engine::SessionCommand command; command.kind = engine::SessionCommand::Kind::memory_budget; command.bytes = bytes;
            set_budget(command);
        }
        impl.budget_->decode(impl.source_.get());
        impl.ready_ = true;
    } catch (...) {
        const auto failure = std::current_exception();
        try { unload_model(); }
        catch (const std::exception& error) { std::cerr << "runtime_load_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "runtime_load_cleanup error=unknown\n"; }
        std::rethrow_exception(failure);
    }
}

void Engine::rebuild(EngineParts&& parts) {
    unload_model();
    model_ = std::move(parts.model);
    executor_ = std::move(parts.executor);
    impl_ = std::move(parts.impl);
    install();
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
using namespace tyloi::cuda;

EngineInfo Engine::describe() const { return impl_->info_; }

bool Engine::ready() const { return impl_->ready_; }

std::unique_ptr<tyloi::Executor::Scope> Engine::enter() { return executor_->enter(); }

void Engine::prepare_request(Request& request, bool batched) {
    engine::prepare_request(request, impl_->text_.get(), impl_->info_, batched, false);
    if (!impl_->idle_state_) {
        request.output.reset();
        request.input = {};
        throw std::logic_error("native engine state is already in use");
    }
    request.state = std::move(impl_->idle_state_);
}

void Engine::retire_request(Request& request) noexcept {
    request.generation = {};
    if (impl_ && request.state) impl_->idle_state_ = std::move(request.state);
    else request.state.reset();
}

SessionResult Engine::session_command(const SessionCommand& command) {
    if (!impl_->ready_) throw std::runtime_error("CUDA engine is unloaded");
    auto scope = executor_->enter();
    return command.kind == engine::SessionCommand::Kind::memory_budget
        ? set_budget(command) : impl_->cache_->control(command);
}

void Engine::drain() {
    auto scope = executor_->enter();
    std::exception_ptr failure;
    try { impl_->weights_->drain(); }
    catch (...) { failure = std::current_exception(); }
    try { executor_->drain(); }
    catch (...) { if (!failure) failure = std::current_exception(); }
    if (failure) std::rethrow_exception(failure);
}

void Engine::unload_model() {
    if (env_frozen() && env_flag("TYLOINFER_CUDA_ENV_DUMP", false))
        std::cerr << "cuda knobs in use:\n" << env_dump();
    impl_->ready_ = false;
    std::exception_ptr failure;
    const auto remember = [&](std::exception_ptr error) { if (!failure) failure = std::move(error); };
    impl_->text_.reset();
    if (executor_) {
        try { drain(); }
        catch (...) { remember(std::current_exception()); }
        std::unique_ptr<tyloi::Executor::Scope> scope;
        try { scope = executor_->enter(); }
        catch (...) { remember(std::current_exception()); }
        clear_bindings();
        impl_->graph_.reset();
        try { executor_->clear(); }
        catch (...) { remember(std::current_exception()); }
        impl_->idle_state_.reset(); model_.reset();
        impl_->weights_->clear_workspace();
        impl_->budget_.reset(); impl_->cache_.reset();
        impl_->weights_.reset();
        scope.reset();
        executor_.reset();
    }
    impl_->ledger_.reset(); impl_->source_.reset();
    if (failure) std::rethrow_exception(failure);
}

ControlResult Engine::control_command(ControlRequest request) {
    if (!impl_->ready_) throw std::runtime_error("engine is unloaded");
    auto scope = executor_->enter();
    return std::visit([&](auto&& value) -> engine::ControlResult {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, engine::DecodeTokens> || std::is_same_v<T, engine::CountTokens> ||
                      std::is_same_v<T, engine::PrepareDuplex> || std::is_same_v<T, engine::PrepareDuplexStep>) {
            if (!impl_->text_) throw std::invalid_argument("text control requires a tokenizer");
            if constexpr (std::is_same_v<T, engine::DecodeTokens>) return impl_->text_->decode_tokens(value.tokens, value.excluded);
            else if constexpr (std::is_same_v<T, engine::CountTokens>) return engine::TokenCount{impl_->text_->count_tokens(value.prompt), impl_->info_.max_context};
            else if constexpr (std::is_same_v<T, engine::PrepareDuplex>) {
                impl_->text_->prepare_duplex_session(value.prompt, value.parameters); return std::move(value.parameters);
            } else { impl_->text_->prepare_duplex_step(value.text, value.input); return std::move(value.input); }
        } else if constexpr (std::is_same_v<T, engine::RuntimeMetrics>) {
            return engine_detail::runtime_metrics(model_->metrics(), {}, std::nullopt, false, nullptr,
                nullptr, 3, nullptr, impl_->source_.get(),
                *impl_->ledger_, *impl_->weights_, nullptr, nullptr, *impl_->cache_, *impl_->budget_,
                executor_->memory(impl_->devices_), Device(kCUDA, impl_->forward_options_.primary), impl_->budget_supported_);
        } else throw std::invalid_argument("model has no duplex component");
    }, std::move(request));
}

bool Engine::can_batch(const EngineRequest&) const { return false; }

bool Engine::exclusive() const { return false; }

void Engine::execute(const std::vector<RequestId>&) {}

void Engine::clear_bindings() noexcept {
    if (impl_->budget_) impl_->budget_->unbind();
}

void Engine::reserve(models::State& state, std::int64_t tokens) {
    std::size_t bytes = 0;
    const Device primary(kCUDA, impl_->forward_options_.primary);
    for (const auto& [device, growth] : state.growth(tokens, impl_->forward_options_))
        if (device == primary) bytes = prefill_budget_add(bytes, growth);
    if (bytes)
        impl_->budget_->reserve_kv(prefill_budget_add(bytes, std::size_t(64) << 20));
}

SessionResult Engine::set_budget(const SessionCommand& command) {
    if (!impl_->budget_supported_)
        throw std::invalid_argument("online memory budgeting requires a single-device CUDA worker");
    auto change = impl_->budget_->change(command);
    impl_->weights_->settle();
    impl_->graph_->invalidate();
    if (impl_->ready_) reset_states(1);
    if (impl_->idle_state_) impl_->idle_state_->reset(1);
    return change.apply();
}

#endif

} // namespace tyloi::engine
