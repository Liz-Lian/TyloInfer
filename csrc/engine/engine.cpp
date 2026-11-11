#include "engine/engine.h"
#include "engine/scheduler/scheduler.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <utility>

#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/resource_metrics.h"
#include "engine/models/prepare.h"
#include "engine/models/source.h"
#include "engine/scheduler/request.h"
#include "engine/budget.h"
#include "engine/generation/prefill_budget.h"
#include "engine/generation/mtp/metrics.h"
#include "engine/generation/batch.h"
#include "engine/generation/batch_ops.h"
#include "engine/models/common/mtp.h"
#include "engine/requests/processor.h"
#include "engine/weights/memory.h"
#include "engine/weights/cache.h"
#include "engine/state/session/sessions.h"
#include <stdexcept>
#include <type_traits>
#endif

namespace tyloi::engine {

Engine::Engine() = default;

void Engine::reset_states(std::int64_t batch) {
    for (auto& [id, request] : scheduler_->requests_) {
        if (!request.state) continue;
#ifdef TYLOINFER_NATIVE_CUDA
        auto* predictor = model_ ? model_->mtp() : nullptr;
        if (predictor && (!request.state->predictor || !request.state->predictor->belongs_to(*predictor)))
            throw std::invalid_argument("predictor state is unavailable or belongs to another model");
        if (budget_) {
            for (const auto& [device, bytes] : request.state->reset_growth(batch))
                if (bytes) budget_->reserve_kv(device, prefill_budget_add(bytes, std::size_t(64) << 20));
        }
#endif
        request.state->release_graphs();
        request.state->reset(batch);
#ifdef TYLOINFER_NATIVE_CUDA
        if (predictor) {
            if (budget_)
                for (const auto& [device, bytes] : predictor->reset_growth(*request.state->predictor, batch))
                    if (bytes) budget_->reserve_kv(device, prefill_budget_add(bytes, std::size_t(64) << 20));
            predictor->reset(*request.state->predictor, batch);
        }
#endif
    }
}

#ifdef TYLOINFER_NATIVE_CUDA
Engine::Engine(std::unique_ptr<models::CausalModel> model, std::unique_ptr<tyloi::Executor> executor,
        Config config, std::shared_ptr<const ModelSource> source,
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
        if (auto* predictor = model_->mtp(); predictor &&
            (!idle_state_->predictor || !idle_state_->predictor->belongs_to(*predictor)))
            throw std::invalid_argument("predictor state is unavailable or belongs to another model");
        info_ = {};
        auto& info = info_;
        info.max_requests = std::max<std::size_t>(1, config_.continuous_batch.max_sequences);
        info.max_context = model_info_.max_context;
        info.vocab_size = static_cast<std::int32_t>(model_info_.vocab_size);
        info.model_type = model_info_.model_type;
        info.stop_token_ids = config_.stop_token_ids;
        info.capabilities = model_->capabilities();
        const auto& caps = info.capabilities;
        info.duplex = caps.full_duplex;
        info.multimodal = caps.image_input || caps.video_input || caps.audio_input || caps.audio_output || caps.full_duplex;
        info.reload = true;
        if (text_) info.chat = text_->chat_template_capabilities();
        auto scope = executor_->enter();
        tyloi::weights::LoadScope memory_load(*ledger_);
        graph_.emplace(model_info_.max_context, executor_->device(), std::move(graph));
        budget_.emplace(*ledger_, *weights_, *cache_, *executor_);
        budget_->bind(model_->mtp() != nullptr);
        if (config_.continuous_batch.max_sequences) {
            batch_ = engine::make_batch(*model_, *executor_, *weights_, *budget_, config_);
        }
        if (const auto bytes = config_.resident_budget_bytes) {
            engine::SessionCommand command; command.kind = engine::SessionCommand::Kind::memory_budget; command.bytes = bytes;
            set_budget(command);
        }
        models::DecodeOptions warmup;
        warmup.routed = weights_->decode() == ::tyloi::weights::Cache::Decode::routed;
        if (config_.decode_graph.enabled && model_info_.decode_graph)
            engine::warmup(*model_, *idle_state_, *executor_, *weights_, warmup);
        weights_->prepare(::tyloi::weights::Cache::Stage::loaded, executor_->device(), source_.get());
        ready_ = true;
    } catch (...) {
        const auto failure = std::current_exception();
        if (graph) {
            try {
                auto scope = executor_->enter();
                graph_.emplace(0, executor_->device(), std::move(graph));
            } catch (...) { graph.reset(); }
        }
        try { unload_model(); }
        catch (const std::exception& error) { std::cerr << "runtime_load_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "runtime_load_cleanup error=unknown\n"; }
        std::rethrow_exception(failure);
    }
}

void Engine::rebind(std::unique_ptr<models::CausalModel> model, std::unique_ptr<tyloi::Executor> executor,
        Config config, std::shared_ptr<const ModelSource> source,
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
    config_ = std::move(config);
    source_ = std::move(source);
    ledger_ = std::move(memory);
    weights_ = std::move(weights);
    idle_state_ = std::move(model_state);
    cache_ = std::move(sessions);
    text_ = std::move(text);
    forward_options_ = std::move(forward);
    mtp_state_ = {};
    if (failure) {
        try {
            auto scope = executor_->enter();
            graph_.emplace(0, executor_->device(), std::move(graph));
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
#ifdef TYLOINFER_NATIVE_CUDA
    if (scheduler_ || executor_) {
        try { shutdown(); }
        catch (const std::exception& error) { std::cerr << "runtime_unload_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "runtime_unload_cleanup error=unknown\n"; }
    }
#endif
}

#ifndef TYLOINFER_NATIVE_CUDA
void Engine::retire_request(Request& request) noexcept {
    request.generation = {};
    request.state.reset();
}
#else
EngineInfo Engine::describe() const { return info_; }

bool Engine::ready() const { return ready_; }

std::unique_ptr<tyloi::Executor::Scope> Engine::enter() { return executor_->enter(); }

void Engine::prepare_request(Request& request, bool batched) {
    engine::prepare_request(request, text_.get(), info_, batched, model_->mtp() != nullptr);
    if (batched) return;
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
        batch_.reset();
        graph_.reset();
        idle_state_.reset(); model_.reset();
        try { executor_->clear(); }
        catch (...) { remember(std::current_exception()); }
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
        } else if constexpr (std::is_same_v<T, engine::GetMetrics>) {
            Metrics predictor;
            if (model_->mtp()) mtp::append_generation_metrics(predictor, mtp_state_.stats);
            const auto* state = idle_state_.get();
            if (!state) {
                for (const auto& [id, active] : scheduler_->requests_) {
                    if (active.state) { state = active.state.get(); break; }
                }
            }
            return resource_metrics(model_->metrics(), {}, state ? state->metrics() : std::nullopt, model_info_.streamed_kv,
                batch_ ? batch_->metrics() : Metrics{}, executor_->metrics(), predictor, source_.get(),
                *ledger_, *weights_, *cache_, *budget_,
                executor_->memory(config_.devices), executor_->device(), config_.online_budget);
        } else throw std::invalid_argument("model has no duplex component");
    }, std::move(request));
}

bool Engine::can_batch(const EngineRequest& request) const {
    return batch_ && !request.input.media && request.input.cache_plan.session_id.empty() &&
        !(request.input.sampling.enable_mtp && model_->mtp());
}

bool Engine::exclusive() const { return false; }

void Engine::execute(const std::vector<RequestId>& eligible) {
    if (batch_) batch_->step(eligible);
}

void Engine::clear_bindings() noexcept {
    if (budget_) budget_->unbind();
}

SessionResult Engine::set_budget(const SessionCommand& command) {
    if (!config_.online_budget)
        throw std::invalid_argument("online memory budgeting requires a single-device CUDA worker");
    auto change = budget_->change(command);
    weights_->settle();
    graph_->invalidate();
    if (batch_) batch_->release_idle_buffers();
    if (ready_) reset_states(1);
    if (idle_state_) {
        idle_state_->release_graphs();
        auto* predictor = model_->mtp();
        if (predictor && (!idle_state_->predictor || !idle_state_->predictor->belongs_to(*predictor)))
            throw std::invalid_argument("predictor state is unavailable or belongs to another model");
        for (const auto& [device, bytes] : idle_state_->reset_growth(1))
            if (bytes) budget_->reserve_kv(device, prefill_budget_add(bytes, std::size_t(64) << 20));
        idle_state_->reset(1);
        if (predictor) {
            for (const auto& [device, bytes] : predictor->reset_growth(*idle_state_->predictor, 1))
                if (bytes) budget_->reserve_kv(device, prefill_budget_add(bytes, std::size_t(64) << 20));
            predictor->reset(*idle_state_->predictor, 1);
        }
    }
    return change.apply();
}

#endif

} // namespace tyloi::engine
