#include "engine/engine.h"
#include "engine/scheduler/scheduler.h"

#include <exception>
#include <iostream>
#include <utility>

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/impl.h"
#include "engine/drain.h"
#include "engine/generate.h"
#include "engine/generation/generation.h"
#include "engine/runtime_metrics.h"
#include "engine/models/prepare.h"
#include "engine/scheduler/request.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/backends/cuda/executor/options.h"
#include <stdexcept>
#include <type_traits>
#endif

namespace tyloi::engine {

Engine::Engine() = default;
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
namespace {
cuda::Executor& cuda_root(tyloi::Executor& executor) { return static_cast<cuda::Executor&>(executor); }
}

template <class F> decltype(auto) Engine::with_cuda(F&& work) {
    if (!impl_->ready_) throw std::runtime_error("CUDA engine is unloaded");
    auto scope = impl_->forward_->enter();
    return std::forward<F>(work)();
}

EngineInfo Engine::describe() const { return impl_->info_; }

bool Engine::ready() const { return impl_->ready_; }

std::unique_ptr<tyloi::Executor::Scope> Engine::enter() { return impl_->forward_->enter(); }

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
    return with_cuda([&] {
        return command.kind == engine::SessionCommand::Kind::memory_budget
            ? set_budget(command) : impl_->cache_->control(command);
    });
}

void Engine::unload_model() {
    if (env_frozen() && env_flag("TYLOINFER_CUDA_ENV_DUMP", false))
        std::cerr << "cuda knobs in use:\n" << env_dump();
    impl_->ready_ = false;
    std::exception_ptr failure;
    const auto remember = [&](std::exception_ptr error) { if (!failure) failure = std::move(error); };
    impl_->text_.reset();
    if (executor_) {
        try { drain_cuda(cuda_root(*executor_), *impl_->weights_, impl_->offload_.get(), impl_->load_.layer_placement, impl_->load_.tensor_parallel, impl_->load_.expert_parallel); }
        catch (...) { remember(std::current_exception()); }
        const auto release = [&] {
            impl_->expert_space_.reset();
            clear_bindings();
            impl_->budget_.reset(); impl_->cache_.reset(); impl_->graph_.reset();
            impl_->forward_.reset(); impl_->idle_state_.reset(); model_.reset(); impl_->mtp_state_ = {};
            cuda_root(*executor_).pipelines().clear();
            impl_->offload_.reset(); impl_->kv_.reset(); impl_->weights_.reset();
        };
        try {
            auto scope = cuda_root(*executor_).enter(impl_->load_.layer_placement.primary_device());
            release();
        } catch (...) {
            remember(std::current_exception());
            release();
        }
        executor_.reset();
    }
    impl_->ledger_.reset(); impl_->source_.reset();
    if (failure) std::rethrow_exception(failure);
}

ControlResult Engine::control_command(ControlRequest request) {
    if (!impl_->ready_) throw std::runtime_error("engine is unloaded");
    return with_cuda([&]() -> engine::ControlResult {
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
                    impl_->mtp_state_, 3, impl_->expert_space_ ? &*impl_->expert_space_ : nullptr, nullptr, impl_->source_.get(),
                    *impl_->ledger_, *impl_->weights_, impl_->kv_.get(), impl_->offload_.get(), *impl_->cache_, *impl_->budget_,
                    impl_->load_.layer_placement, impl_->load_.tensor_parallel, impl_->load_.expert_parallel);
            } else throw std::invalid_argument("model has no duplex component");
        }, std::move(request));
    });
}

bool Engine::can_batch(const EngineRequest&) const { return false; }

bool Engine::exclusive() const { return false; }

void Engine::execute(const std::vector<RequestId>&) {}

tyloi::StepSequence<Engine::Prepared> Engine::prepare(InferenceRequest& input, models::State& state) {
    return model_->prepare(input.prompt, *input.vision, state);
}

std::pair<tyloi::StepResult<Engine::Prepared>, double> Engine::advance_preparation(tyloi::StepSequence<Prepared>& preparation) {
    internal::PrefillCudaTimer timer;
    auto step = preparation.next();
    TYLOINFER_CUDA_CHECK(cudaEventRecord(timer.finished_event(), mfq_get_current_cuda_stream()));
    return std::pair{std::move(step), timer.elapsed_ms()};
}

tyloi::StepSequence<EventData> Engine::generate_text(InferenceRequest& input, InferenceOutput& output, models::State& state, std::optional<Prepared> prepared, const RequestId& id, bool) {
    return internal::generate(*model_, state, impl_->forward_.get(), cuda_root(*executor_).collectives(),
        cuda_root(*executor_).forward_mode(impl_->load_.layer_placement.primary_device()), cuda_root(*executor_).profiler(), *impl_->weights_,
        cuda_root(*executor_).mode().ffn.serial, impl_->load_.report_cuda_memory,
        impl_->load_.dsv4_cpu_offload_layers.empty() && impl_->load_.dense_cpu_layer_count == 0 &&
            model_parallel_cuda_graph_enabled(impl_->load_, cuda_root(*executor_), *impl_->weights_),
        [this] { drain_cuda(cuda_root(*executor_), *impl_->weights_, impl_->offload_.get(), impl_->load_.layer_placement, impl_->load_.tensor_parallel, impl_->load_.expert_parallel); },
        impl_->model_info_, impl_->source_.get(), *impl_->graph_, *impl_->cache_, impl_->runtime_, input, output, impl_->mtp_state_,
        static_cast<MtpModule*>(nullptr), std::move(prepared), nullptr, id, impl_->restore_);
}

tyloi::StepSequence<EventData> Engine::generate(const RequestId& id, Request& request) {
    return engine::generate_prepared(*this, *request.state, request.input, *request.output, id, request.batched);
}

void Engine::restore_graph_expert_space() {
    if (impl_->weights_->tq) return;
    prepare_moe_decode_residency(impl_->weights_->moe);
    default_context(impl_->load_.layer_placement.primary_device())->reclaim_idle_memory();
    if (impl_->expert_space_) impl_->expert_space_->restore(runtime_options::workspace_reserve_bytes(), impl_->source_.get());
}

void Engine::clear_bindings() noexcept {
    impl_->restore_ = {};
    if (impl_->kv_) impl_->kv_->set_reservation({});
    set_moe_workspace_reservation(impl_->weights_->moe, {});
    if (impl_->offload_) impl_->offload_->set_ram_reservation({});
}

SessionResult Engine::set_budget(const SessionCommand& command) {
    if (impl_->load_.layer_placement.devices.size() > 1 || impl_->load_.tensor_parallel.enabled() || impl_->load_.expert_parallel.enabled())
        throw std::invalid_argument("online memory budgeting requires a single-device CUDA worker");
    return impl_->budget_->set(command, [this] {
        if (impl_->expert_space_) impl_->expert_space_->release();
        finish_moe_expert_exchanges(impl_->weights_->moe);
        impl_->graph_->invalidate();
        if (impl_->ready_) reset_states(1);
        if (impl_->idle_state_) impl_->idle_state_->reset(1);
    });
}

#endif

} // namespace tyloi::engine
