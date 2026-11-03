#pragma once

#include "engine/engine.h"
#include "engine/drain.h"
#include "engine/generation/generation.h"
#include "engine/generation/mtp/state.h"
#include "engine/load.h"
#include "engine/runtime_metrics.h"
#include "engine/backends/cuda/native/scope.h"
#include "engine/budget.h"
#include "engine/requests/processor.h"
#include "engine/requests/output.h"
#include "engine/models/prepare.h"
#include "engine/models/causal_model.h"
#include "engine/models/common/causal_metadata.h"
#include "engine/models/common/components.h"
#include "engine/scheduler/table.h"
#include "engine/weights/workspace.h"
#include "engine/generation/batch.h"
#include "engine/generate.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/executor/execution.h"
#include "engine/state/kv_memory.h"
#include "engine/state/kv_offload.h"
#include "engine/models/common/session/cache.h"
#include "engine/models/common/model_config.h"
#include "engine/weights/loader.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/weights/moe/kv_expert_loan.h"
#include "engine/weights/routed_cache.h"
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

namespace mfq::cuda {

template <class Model, class State, class Parts, auto Load>
class CudaEngine final : public engine::Engine {
    using Options = CudaEngineOptions;

public:
    using Prepared = PreparedPrompt;

    CudaEngine(
        CudaEngineOptions options,
        std::shared_ptr<const mfq::ModelSource> source)
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
    std::unique_ptr<mfq::Executor::Scope> enter() override {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        return executor_->enter();
#else
        return executor_->enter(execution_->config.layer_placement.primary_device());
#endif
    }
    std::unique_ptr<engine::ExecutionRequest> prepare_request(engine::EngineRequest request, bool batched) override {
        return engine::prepare_request(std::move(request), text_.get(), info_, batched, mtp_available());
    }

    engine::SessionResult session_command(
        const engine::SessionCommand& command) override {
        return with_cuda([&] {
            if (command.kind != engine::SessionCommand::Kind::memory_budget)
                return engine::control_session(*cache_, command);
            return set_budget(command);
        });
    }

    std::int64_t reload_model(std::int64_t context) override {
        if (context < 1) throw std::invalid_argument("reload context must be positive");
        unload_model();
        options_.context_size = context;
        source_ = mfq::open_model_source(options_.model_path);
        load();
        return info_.max_context;
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
                expert_space_.reset();
                clear_bindings();
                batching_.reset();
                budget_.reset();
                cache_.reset();
                graph_.reset();
                components_.reset();
                executor_.reset();
                state_.reset();
                model_.reset();
                mtp_state_ = {};
                execution_->executor().pipelines().clear();
                offload_.reset();
                kv_.reset();
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

    mfq::StepSequence<Prepared> prepare(engine::InferenceRequest& input) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if constexpr (requires { static_cast<const mfq::models::CausalModel&>(language()).prepare(
                                   input.prompt, *input.vision, state()); }) {
            const mfq::models::CausalModel& causal = language();
            return causal.prepare(input.prompt, *input.vision, state());
        } else
#endif
        return components().prepare(
            language(), execution_->executor().projector(execution_->executor().mode().ffn, model_parallel_primary_device(execution_->config)), weights_.moe,
            execution_->executor().collectives(), execution_->config.layer_placement.primary_device(),
            executor_.get(), input.prompt, *input.vision);
    }

    auto advance_preparation(mfq::StepSequence<Prepared>& preparation) {
        internal::PrefillCudaTimer timer;
        auto step = preparation.next();
        TYLOINFER_CUDA_CHECK(cudaEventRecord(
            timer.finished_event(), mfq_get_current_cuda_stream()));
        return std::pair{std::move(step), timer.elapsed_ms()};
    }

    engine::Generation generate_text(
        engine::InferenceRequest& input, engine::InferenceOutput& output,
        std::optional<Prepared> prepared, const engine::RequestId& id, bool batched) {
        return internal::generate(
            language(), state(), executor_.get(), execution_->executor().collectives(),
            execution_->executor().forward_mode(execution_->config.layer_placement.primary_device()), execution_->executor().profiler(), weights_,
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
            batched ? batching_.get() : nullptr, id, restore_decode_workspace_);
    }

    engine::Generation generate(
        const engine::RequestId& id, engine::ExecutionRequest& request) override {
        if (request.batched) graph_->invalidate();
        return engine::generate_prepared(
            *this, request.input, request.output, id, request.batched);
    }

private:
    Options options_;
    std::shared_ptr<KvMemoryManager> kv_;
    RoutedCache weights_;
    std::shared_ptr<KvOffloadStore> offload_;
    std::unique_ptr<Executor> root_executor_;
    std::unique_ptr<CudaExecutionContext> execution_;
    std::shared_ptr<WeightMemoryLedger> weight_memory_;
    std::shared_ptr<const mfq::ModelSource> source_;
    CudaRuntimeConfig config_;
    std::shared_ptr<Model> model_;
    std::unique_ptr<State> state_;
    std::unique_ptr<Executor> executor_;
    std::optional<Parts> components_;
    mfq::models::Info model_info_{};
    mfq::engine::mtp::State mtp_state_;
    std::optional<DecodeGraphCache> graph_;
    std::optional<internal::TextSessionCache> cache_;
    std::optional<Budget> budget_;
    std::unique_ptr<engine::Batch> batching_;
    std::optional<ExpertSpace> expert_space_;
    std::function<void()> restore_decode_workspace_;
    std::unique_ptr<engine::TextProcessor> text_;
    engine::EngineInfo info_;
    bool ready_ = false;
    bool duplex_active_ = false;
    bool preparing_ = false;

    template <class F>
    decltype(auto) with_cuda(F&& function) {
        if (!ready_ || !execution_ || !model_)
            throw std::runtime_error("CUDA engine is unloaded");
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        auto scope = executor_->enter();
#else
        CudaScope scope(execution_->config.layer_placement.primary_device(), execution_->executor().contexts());
#endif
        mfq_tensor_backend::NoGradGuard no_grad;
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

    void restore_graph_expert_space() {
        if (weights_.tq || preparing_) return;
        prepare_moe_decode_residency(weights_.moe);
        default_context(execution_->config.layer_placement.primary_device())->reclaim_idle_memory();
        if (expert_space_)
            expert_space_->restore(runtime_options::workspace_reserve_bytes(), source_.get());
    }

    void clear_bindings() noexcept {
        restore_decode_workspace_ = {};
        if (kv_) kv_->set_reservation({});
        set_moe_workspace_reservation(weights_.moe, {});
        if (offload_) offload_->set_ram_reservation({});
    }

    engine::SessionResult set_budget(const engine::SessionCommand& command) {
        if (!online_budget_supported())
            throw std::invalid_argument(
                "online memory budgeting requires a single-device CUDA worker");
        return budget_->set(command, [this] {
            if (expert_space_) expert_space_->release();
            finish_moe_expert_exchanges(weights_.moe);
            graph_->invalidate();
            if (batching_) batching_->release_idle_buffers();
            if (components().mtp) components().mtp->reset(1);
            state().reset(1);
        });
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

            auto [execution_config, numeric_options] = mfq::cuda::load_cuda_execution_config();
            root_executor_ = std::make_unique<Executor>(std::move(numeric_options), 0);
            execution_ = std::make_unique<CudaExecutionContext>(
                *root_executor_, std::move(execution_config), weights_, kv_, offload_);
            CudaScope outer_scope(
                execution_->config.layer_placement.primary_device(), execution_->executor().contexts());
            internal::setup_cuda_load(options_, execution_->config, *root_executor_, weights_, kv_, offload_);
            execution_->executor().profiler().enabled = false;
            weight_memory_ = std::make_shared<WeightMemoryLedger>();
            WeightMemoryLoadScope memory_load(*weight_memory_);
            mfq_tensor_backend::NoGradGuard no_grad;
            config_ = internal::resolve_cuda_runtime_config(options_);

            const auto kv_options = runtime_options::qsa_kv_offload();
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
            weight_loader::Loader loader{
                *source_, execution_->config.weights,
                {execution_->config.layer_placement.primary_device(), false,
                 execution_->config.drop_file_cache, execution_->config.tensor_parallel,
                 moe_parallel_config(execution_->config)},
                weights_, execution_->executor().tq()};
            executor_ = execution_->executor().fork(model_parallel_primary_device(execution_->config));
            const bool preallocate = runtime_options::kv_vram_preallocated();
            model_ = Load(
                source_, loader, execution_->config.layer_placement, execution_->config.yarn,
                execution_->config.dense_cpu_layer_count, options_,
                execution_->config.n_gpu_layers, execution_->config.yarn_context_extension,
                execution_->config.expert_parallel.enabled(),
                execution_->config.tensor_parallel_mirror_linear_attention_scalars);
            state_ = std::make_unique<State>(*model_, execution_->executor().mode().transposed);
            if (preallocate) state_->preallocate();
            components_.emplace(Parts::load(
                *model_, loader, execution_->config.layer_placement, execution_->config.dense_cpu_layer_count,
                *executor_, execution_->executor().collectives(),
                execution_->executor().mmq().kv_cache_capacity,
                execution_->executor().mode().trace, execution_->executor().mode().trace_layer,
                true, !execution_->config.dsv4_cpu_offload_layers.empty(), preallocate));
            model_info_ = mfq::models::model_info(language());
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
            execution_->config.layer_placement.primary_device());
        if (offload_) {
            offload_->set_ram_reservation([this](std::size_t bytes) {
                cache_->reserve_ram(bytes);
                weights_.reserve(bytes);
            });
        }
        if (config_.continuous_batch.max_sequences) {
            auto batch_config = config_.continuous_batch;
            batch_config.cuda_graph &= model_parallel_cuda_graph_enabled(execution_->config, execution_->executor(), execution_->weights);
            batching_ = make_batch(
                std::shared_ptr<const Model>(model_), &state(),
                execution_->executor().fork(model_parallel_primary_device(execution_->config)), execution_->executor().collectives(),
                execution_->executor().forward_mode(execution_->config.layer_placement.primary_device()), weights_.moe, std::move(batch_config),
                config_.generation.prefill_chunk_size,
                runtime_options::kv_vram_preallocated(),
                execution_->config.dense_cpu_layer_count != 0 ||
                    !execution_->config.dsv4_cpu_offload_layers.empty(),
                execution_->config.weights.min_slots);
        }
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if (weights_.moe) {
            auto expert_loan = kv_ ? kv_->expert_loan() : nullptr;
            if (!expert_loan && (components().mtp ||
                                 moe_expert_cache_ssd_experts(weights_.moe))) {
                expert_loan = std::make_shared<KvExpertLoan>();
                set_moe_kv_expert_loan(weights_.moe, expert_loan);
            }
            if (expert_loan) {
                expert_space_.emplace(
                    weights_, execution_->config.layer_placement.primary_device(),
                    std::move(expert_loan));
            }
            restore_decode_workspace_ = [this] { restore_graph_expert_space(); };
            set_moe_workspace_reservation(
                weights_.moe,
                [this](std::size_t bytes) {
                    budget_->reclaim(
                        bytes, expert_space_ ? &*expert_space_ : nullptr);
                },
                [this](std::size_t bytes) {
                    cache_->reserve_ram(bytes);
                    weights_.reserve(bytes);
                });
        }
#endif
        if (const auto bytes = runtime_options::resident_budget_bytes()) {
            engine::SessionCommand command;
            command.kind = engine::SessionCommand::Kind::memory_budget;
            command.bytes = bytes;
            set_budget(command);
        }

        preparing_ = false;
        restore_graph_expert_space();

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
        info_.stop_token_ids = mfq::models::stop_token_ids(
            mfq::models::config_json(*source_, options_.config_path), info_.vocab_size);
    }

    template <class T>
    engine::ControlResult control_model(T value) {
        if constexpr (std::is_same_v<T, engine::RuntimeMetrics>) {
            const auto& model = std::as_const(language());
            const auto& parts = std::as_const(components());
            const auto component_metrics = [&] {
                if constexpr (requires { static_cast<const mfq::models::CausalModel&>(model); })
                    return static_cast<const mfq::models::CausalModel&>(model).metrics();
                else
                    return parts.metrics();
            }();
            engine::Metrics profile_metrics;
            if constexpr (requires { parts.profile_metrics(model); })
                profile_metrics = parts.profile_metrics(model);
            std::optional<engine::Metrics> memory;
            if constexpr (requires { parts.memory_metrics(model); })
                memory = parts.memory_metrics(model);
            const bool streamed_kv = parts.streamed_kv();
            const auto* mtp = parts.mtp.get();
            return engine_detail::runtime_metrics(
                component_metrics, profile_metrics, std::move(memory), streamed_kv, mtp,
                mtp_state_, 3,
                expert_space_ ? &*expert_space_ : nullptr, batching_.get(),
                source_.get(), *weight_memory_, weights_, kv_.get(),
                offload_.get(), *cache_, *budget_,
                execution_->config.layer_placement, execution_->config.tensor_parallel,
                execution_->config.expert_parallel);
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

} // namespace mfq::cuda
