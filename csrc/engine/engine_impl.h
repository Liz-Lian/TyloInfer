#pragma once

#include "engine/engine.h"
#include "engine/drain.h"
#include "engine/generation/generation.h"
#include "engine/generation/mtp/metrics.h"
#include "engine/generation/mtp/state.h"
#include "engine/load.h"
#include "engine/resource_metrics.h"
#include "engine/backends/cuda/native/scope.h"
#include "engine/budget.h"
#include "engine/requests/processor.h"
#include "engine/requests/output.h"
#include "engine/models/prepare.h"
#include "engine/models/causal_model.h"
#include "engine/models/metadata.h"
#include "engine/models/common/components.h"
#include "engine/models/common/mtp.h"
#include "engine/models/common/prompt.h"
#include "engine/scheduler/request.h"
#include "engine/generation/batch.h"
#include "engine/generation/cuda.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/executor/execution.h"
#include "engine/state/kv_memory.h"
#include "engine/state/kv_offload.h"
#include "engine/models/session/cache.h"
#include "engine/models/source/model_config.h"
#include "engine/weights/loader.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/weights/memory.h"
#include "engine/models/source.h"

#include <algorithm>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace tyloi::cuda {

template <class Model, class State, class Parts, auto Load>
class CudaEngine final : public engine::Engine {
    using Options = CudaEngineOptions;

public:
    using Prepared = tyloi::models::Prepared;

    CudaEngine(
        CudaEngineOptions options,
        std::shared_ptr<const tyloi::ModelSource> source)
        : options_(std::move(options)), source_(std::move(source)) {
        if (!source_) throw std::invalid_argument("CUDA engine requires a model source");
        load();
        start();
    }

    ~CudaEngine() override {
        try {
            shutdown();
        } catch (const std::exception& error) {
            std::cerr << "runtime_unload_cleanup error=" << error.what() << '\n';
        }
    }

protected:
    engine::EngineInfo describe() const override { return info_; }

    bool ready() const override { return ready_; }
    std::unique_ptr<tyloi::Executor::Scope> enter() override {
#ifdef TYLOINFER_NATIVE_CUDA
        return executor_->enter();
#else
        return executor_->enter(execution_->config.layer_placement.primary_device());
#endif
    }
    void prepare_request(engine::Request& request, bool batched) override {
        engine::prepare_request(request, text_.get(), info_, batched, mtp_available());
    }

    engine::SessionResult session_command(
        const engine::SessionCommand& command) override {
        return with_cuda([&] {
            if (command.kind != engine::SessionCommand::Kind::memory_budget)
                return cache_->control(command);
            return set_budget(command);
        });
    }

    void unload_model() override {
        if (env_frozen() && env_flag("TYLOINFER_CUDA_ENV_DUMP", false))
            std::cerr << "cuda knobs in use:\n" << env_dump();
        ready_ = false;
        std::exception_ptr failure;
        const auto remember = [&](std::exception_ptr error) {
            if (!failure) failure = std::move(error);
        };

        text_.reset();

        if (execution_) {
            try {
                drain_cuda(*root_executor_, weights_, offload_.get(),
                           execution_->config.layer_placement, execution_->config.tensor_parallel,
                           execution_->config.expert_parallel);
            } catch (...) {
                remember(std::current_exception());
            }
            {
                CudaScope scope(execution_->config.layer_placement.primary_device(), execution_->executor().contexts());
                clear_bindings();
                batching_.reset();
                graph_.reset();
                components_.reset();
                images_.clear();
                executor_.reset();
                state_.reset();
                model_.reset();
                mtp_state_ = {};
                execution_->executor().pipelines().clear();
                offload_.reset();
                kv_.reset();
                weights_.clear_workspace();
                budget_.reset();
                cache_.reset();
                weights_.reset();
            }
            execution_.reset();
        }
        root_executor_.reset();
        weight_memory_.reset();
        source_.reset();
        preparing_ = false;
        duplex_active_ = false;
        if (failure) std::rethrow_exception(failure);
    }

    engine::ControlResult control_command(engine::ControlRequest request) override {
        if (!ready_) throw std::runtime_error("CUDA engine is unloaded");
        return with_cuda([&] {
            return std::visit([&](auto&& value) -> engine::ControlResult {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, engine::DecodeTokens> ||
                              std::is_same_v<T, engine::CountTokens> ||
                              std::is_same_v<T, engine::PrepareDuplex> ||
                              std::is_same_v<T, engine::PrepareDuplexStep>) {
                    if (!text_)
                        throw std::invalid_argument("text control requires a tokenizer");
                    if constexpr (std::is_same_v<T, engine::DecodeTokens>) {
                        return text_->decode_tokens(value.tokens, value.excluded);
                    } else if constexpr (std::is_same_v<T, engine::CountTokens>) {
                        return engine::TokenCount{
                            text_->count_tokens(value.prompt), info_.max_context};
                    } else if constexpr (std::is_same_v<T, engine::PrepareDuplex>) {
                        text_->prepare_duplex_session(value.prompt, value.parameters);
                        return std::move(value.parameters);
                    } else {
                        text_->prepare_duplex_step(value.text, value.input);
                        return std::move(value.input);
                    }
                } else {
                    return control_model(std::move(value));
                }
            }, std::move(request));
        });
    }

public:
    bool can_batch(const engine::EngineRequest& request) const override {
        return batching_ && engine::batch_compatible(
            request, cache_->persistent_prefix_enabled(), bool(components().mtp));
    }

    bool exclusive() const override { return duplex_active_; }
    bool mtp_available() const { return bool(components().mtp); }

    void execute(const std::vector<engine::RequestId>& eligible) override {
        if (batching_) batching_->step(eligible);
    }

    tyloi::StepSequence<Prepared> prepare(engine::InferenceRequest& input, State& state) {
#ifdef TYLOINFER_NATIVE_CUDA
        if constexpr (requires { static_cast<const tyloi::models::CausalModel&>(language()).prepare(
                                   input.prompt, *input.vision, state); }) {
            const tyloi::models::CausalModel& causal = language();
            return causal.prepare(input.prompt, *input.vision, state);
        } else
#endif
        return components().prepare(
            language(), images_,
#ifndef TYLOINFER_NATIVE_CUDA
            execution_->executor().projector(execution_->executor().mode().ffn, model_parallel_primary_device(execution_->config)),
            execution_->executor().collectives(),
#endif
            execution_->config.layer_placement.primary_device(),
#ifndef TYLOINFER_NATIVE_CUDA
            executor_.get(),
#endif
            input.prompt, *input.vision);
    }

    auto advance_preparation(tyloi::StepSequence<Prepared>& preparation) {
        internal::PrefillCudaTimer timer;
        auto step = preparation.next();
        TYLOINFER_CUDA_CHECK(cudaEventRecord(
            timer.finished_event(), mfq_get_current_cuda_stream()));
        return std::pair{std::move(step), timer.elapsed_ms()};
    }

    engine::Generation generate_text(
        engine::InferenceRequest& input, engine::InferenceOutput& output, State& state,
        std::optional<Prepared> prepared, const engine::RequestId& id, bool batched) {
        bool decode_ready = false;
#ifdef TYLOINFER_NATIVE_CUDA
        decode_ready = !preparing_;
#endif
        return internal::generate(
            language(), state, executor_.get(), execution_->executor().collectives(),
            execution_->executor().forward_mode(execution_->config.layer_placement.primary_device()), execution_->executor().profiler(), weights_, *budget_,
            execution_->executor().mode().ffn.serial,
            execution_->config.report_cuda_memory,
            execution_->config.dsv4_cpu_offload_layers.empty() &&
                execution_->config.dense_cpu_layer_count == 0 &&
                model_parallel_cuda_graph_enabled(execution_->config, execution_->executor(), execution_->weights),
            [this] {
                drain_cuda(*root_executor_, weights_, offload_.get(),
                           execution_->config.layer_placement, execution_->config.tensor_parallel,
                           execution_->config.expert_parallel);
            }, model_info_, source_.get(), *graph_,
            *cache_, config_, input, output,
            mtp_state_,
            batched ? nullptr : components().mtp.get(), std::move(prepared),
            batched ? batching_.get() : nullptr, id, decode_ready);
    }

    engine::Generation generate(
        const engine::RequestId& id, engine::Request& request) override {
        if (request.batched) graph_->invalidate();
        return engine::generate_prepared(*this, state(), request.input, *request.output, id, request.batched);
    }

private:
    friend void ::tyloi::cuda::reload_cuda_engine(engine::Engine&, CudaEngineOptions);
    Options options_;
    std::shared_ptr<KvMemoryManager> kv_;
    ::tyloi::weights::Cache weights_;
    std::shared_ptr<KvOffloadStore> offload_;
    std::unique_ptr<Executor> root_executor_;
    std::unique_ptr<CudaExecutionContext> execution_;
    std::shared_ptr<tyloi::weights::Memory> weight_memory_;
    std::shared_ptr<const tyloi::ModelSource> source_;
    tyloi::engine::Config config_;
    std::shared_ptr<Model> model_;
    std::unique_ptr<State> state_;
    std::unique_ptr<Executor> executor_;
    MediaCache images_;
    std::optional<Parts> components_;
    tyloi::models::Info model_info_{};
    tyloi::engine::mtp::State mtp_state_;
    std::optional<DecodeGraphCache> graph_;
    std::optional<engine::SessionCache<internal::CudaSessionOps>> cache_;
    std::optional<engine::Budget> budget_;
    std::unique_ptr<engine::Batch> batching_;
    std::unique_ptr<engine::TextProcessor> text_;
    engine::EngineInfo info_;
    bool ready_ = false;
    bool duplex_active_ = false;
    bool preparing_ = false;

    template <class F>
    decltype(auto) with_cuda(F&& function) {
        if (!ready_ || !execution_ || !model_)
            throw std::runtime_error("CUDA engine is unloaded");
#ifdef TYLOINFER_NATIVE_CUDA
        auto scope = executor_->enter();
#else
        CudaScope scope(execution_->config.layer_placement.primary_device(), execution_->executor().contexts());
#endif
        ::tyloi::cuda::NoGradGuard no_grad;
        return std::forward<F>(function)();
    }

    Model& language() { return *model_; }
    State& state() { return *state_; }
    Parts& components() { return *components_; }
    const Parts& components() const { return *components_; }

    bool online_budget_supported() const {
        return execution_->config.layer_placement.devices.size() <= 1 &&
            !execution_->config.tensor_parallel.enabled() &&
            !execution_->config.expert_parallel.enabled();
    }

    void clear_bindings() noexcept {
        if (kv_) kv_->bind_budget(nullptr);
        if (budget_) budget_->unbind();
        if (offload_) offload_->bind_budget(nullptr);
    }

    engine::SessionResult set_budget(const engine::SessionCommand& command) {
        if (!online_budget_supported())
            throw std::invalid_argument(
                "online memory budgeting requires a single-device CUDA worker");
        auto change = budget_->change(command);
        weights_.settle();
        graph_->invalidate();
        if (batching_) batching_->release_idle_buffers();
        if (components().mtp) components().mtp->reset(1);
        state().reset(1);
        return change.apply();
    }

    void load() {
        try {
            ready_ = false;
            preparing_ = true;
            const bool dump_env = env_flag("TYLOINFER_CUDA_ENV_DUMP", false);
            env_freeze();
            if (dump_env) std::cerr << "cuda knobs at load:\n" << env_dump();
            if (!options_.context_size) options_.context_size = 32768;
            if (execution_) throw std::logic_error("CUDA load requires retired resources");

            auto [execution_config, numeric_options] = tyloi::cuda::load_cuda_execution_config();
            root_executor_ = std::make_unique<Executor>(std::move(numeric_options), 0);
            execution_ = std::make_unique<CudaExecutionContext>(
                *root_executor_, std::move(execution_config), weights_, kv_, offload_);
            auto outer_scope = root_executor_->enter(
                execution_->config.layer_placement.primary_device());
            internal::setup_cuda_load(options_, execution_->config, *root_executor_, weights_, kv_, offload_);
            execution_->executor().profiler().enabled = false;
            weight_memory_ = std::make_shared<tyloi::weights::Memory>();
            tyloi::weights::LoadScope memory_load(*weight_memory_);
            config_ = internal::resolve_cuda_config(options_);

            const auto kv_options = ::tyloi::cuda::options::qsa_kv_offload();
            kv_ = std::make_shared<KvMemoryManager>();
            if (kv_options.gpu_budget_bytes) {
                kv_.reset();
                KvOffloadConfig offload_config;
                offload_config.gpu_budget_bytes = kv_options.gpu_budget_bytes;
                offload_config.ram_budget_bytes = internal::cuda_has_separate_memory()
                    ? kv_options.ram_budget_bytes : 0;
                offload_config.buffer_bytes = std::min<std::size_t>(
                    offload_config.buffer_bytes, kv_options.gpu_budget_bytes / 4);
                offload_config.directory = kv_options.directory;
                offload_ = std::make_shared<KvOffloadStore>(std::move(offload_config));
            }

            CudaScope load_scope(
                execution_->config.layer_placement.primary_device(), execution_->executor().contexts());
            auto loader = weights_.loader(
                *source_, execution_->config.weights,
                {execution_->config.layer_placement.primary_device(), false,
                 execution_->config.drop_file_cache, execution_->config.tensor_parallel,
                 moe_parallel_config(execution_->config)});
            executor_ = execution_->executor().fork(model_parallel_primary_device(execution_->config));
            const bool preallocate = ::tyloi::cuda::options::kv_vram_preallocated();
            model_ = Load(
                source_, loader, execution_->config.layer_placement, execution_->config.yarn,
                execution_->config.dense_cpu_layer_count, options_,
                execution_->config.n_gpu_layers, execution_->config.yarn_context_extension,
                execution_->config.expert_parallel.enabled(),
                execution_->config.tensor_parallel_mirror_linear_attention_scalars);
            state_ = std::make_unique<State>(*model_, execution_->executor().mode().transposed);
            if (preallocate) state_->preallocate();
            components_.emplace(Parts::load(
                *model_, loader, weights_, execution_->config.layer_placement, execution_->config.dense_cpu_layer_count,
                *executor_, execution_->executor().collectives(),
                execution_->executor().mmq().kv_cache_capacity,
                execution_->executor().mode().trace, execution_->executor().mode().trace_layer,
                true, !execution_->config.dsv4_cpu_offload_layers.empty(), preallocate));
            model_info_ = tyloi::models::model_info(language());
            finalize_kv(kv_.get(), weights_, offload_);
            finish_load(kv_.get(), weights_, 0,
                execution_->config.report_cuda_memory);
            if (offload_ && !components().streamed_kv())
                throw std::invalid_argument("CUDA streamed KV requires a QSA model");
            initialize_runtime();
            ready_ = true;
        } catch (...) {
            const auto failure = std::current_exception();
            try {
                unload_model();
            } catch (const std::exception& error) {
                std::cerr << "runtime_load_cleanup error=" << error.what() << '\n';
            }
            std::rethrow_exception(failure);
        }
    }

    void initialize_runtime() {
        graph_.emplace(model_info_.max_context);
        cache_.emplace(
            config_.session_cache, config_.prefix_cache,
            internal::make_cuda_paged_prefix_cache(
                *source_, model_info_.max_context,
                state().supports_paged_text_session_state(), config_.prefix_cache),
            state().supports_text_session_state(),
            state().supports_text_session_state() ? 0 : 1,
            internal::cuda_has_separate_memory());
        budget_.emplace(
            *weight_memory_, weights_, *cache_,
            tyloi::Device(tyloi::kCUDA, execution_->config.layer_placement.primary_device()));
        if (offload_) offload_->bind_budget(&*budget_);
        if (config_.continuous_batch.max_sequences) {
            auto batch_config = config_.continuous_batch;
            batch_config.cuda_graph &= model_parallel_cuda_graph_enabled(execution_->config, execution_->executor(), execution_->weights);
            batching_ = make_batch(
                std::shared_ptr<const Model>(model_), &state(),
                execution_->executor().fork(model_parallel_primary_device(execution_->config)), execution_->executor().collectives(),
                execution_->executor().forward_mode(execution_->config.layer_placement.primary_device()), *budget_, std::move(batch_config),
                config_.generation.prefill_chunk_size,
                ::tyloi::cuda::options::kv_vram_preallocated(),
                execution_->config.dense_cpu_layer_count != 0 ||
                    !execution_->config.dsv4_cpu_offload_layers.empty(),
                execution_->config.weights.min_slots);
        }
#ifdef TYLOINFER_NATIVE_CUDA
        budget_->bind(bool(components().mtp));
#endif
        if (const auto bytes = ::tyloi::cuda::options::resident_budget_bytes()) {
            engine::SessionCommand command;
            command.kind = engine::SessionCommand::Kind::memory_budget;
            command.bytes = bytes;
            set_budget(command);
        }

        preparing_ = false;
        weights_.prepare(::tyloi::weights::Cache::Stage::loaded,
            tyloi::Device(tyloi::kCUDA, execution_->config.layer_placement.primary_device()), source_.get());

        info_ = {};
        info_.max_requests = std::max<std::size_t>(
            1, config_.continuous_batch.max_sequences);
        info_.max_context = model_info_.max_context;
        info_.vocab_size = static_cast<std::int32_t>(model_info_.vocab_size);
        info_.model_type = model_info_.model_type;
        describe_model(language(), components(), info_);
        info_.reload = true;
        text_ = engine::TextProcessor::load(
            *source_, options_.tokenizer_model, info_.vocab_size, info_.model_type);
        info_.chat = text_->chat_template_capabilities();
        info_.stop_token_ids = tyloi::models::stop_token_ids(
            tyloi::models::config_json(*source_, options_.config_path), info_.vocab_size);
    }

    template <class T>
    engine::ControlResult control_model(T value) {
        if constexpr (std::is_same_v<T, engine::GetMetrics>) {
            const auto& model = std::as_const(language());
            const auto& parts = std::as_const(components());
            const auto component_metrics = [&] {
                if constexpr (requires { static_cast<const tyloi::models::CausalModel&>(model); })
                    return static_cast<const tyloi::models::CausalModel&>(model).metrics();
                else
                    return parts.metrics();
            }();
            engine::Metrics profile_metrics;
            if constexpr (requires { parts.profile_metrics(model,state()); })
                profile_metrics = parts.profile_metrics(model,state());
            std::optional<engine::Metrics> memory;
            if constexpr (requires { parts.memory_metrics(model,state()); })
                memory = parts.memory_metrics(model,state());
            const bool streamed_kv = parts.streamed_kv();
            const auto* mtp = parts.mtp.get();
            const auto batch_metrics = batching_ ? batching_->metrics() : engine::Metrics{};
            const auto kv_metrics = kv_ ? kv_->metrics() : engine::Metrics{};
            const auto offload_metrics = offload_
                ? std::optional<engine::Metrics>{offload_->metrics()} : std::nullopt;
            engine::Metrics predictor_metrics;
            if (mtp) {
                engine::mtp::append_generation_metrics(predictor_metrics, mtp_state_.stats);
                predictor_metrics.emplace_back("mtp_predictor_graph_replays", double(mtp->graph_replay_count()));
            }
            return engine::resource_metrics(
                component_metrics, profile_metrics, std::move(memory), streamed_kv, 3,
                batch_metrics, kv_metrics, offload_metrics, predictor_metrics,
                source_.get(), *weight_memory_, weights_, *cache_, *budget_,
                execution_->executor().memory(cuda_devices(execution_->config)),
                tyloi::Device(tyloi::kCUDA, execution_->config.layer_placement.primary_device()), online_budget_supported());
        } else {
            auto result = components().control(value);
            if constexpr (std::is_same_v<T, MfqDuplexSessionParams>)
                duplex_active_ = true;
            else if constexpr (!std::is_same_v<T, MfqDuplexStepInput>)
                duplex_active_ = false;
            return result;
        }
    }
};

} // namespace tyloi::cuda
