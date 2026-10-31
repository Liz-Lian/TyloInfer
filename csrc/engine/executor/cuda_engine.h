#pragma once

#include "engine.h"
#include "flow.h"
#include "load.h"
#include "metrics.h"
#include "scope.h"
#include "tq.h"
#include "../requests/processor.h"
#include "../requests/table.h"
#include "../weights/workspace.h"
#include "engine/batch/batch.h"
#include "backends/cuda/engine/executor/generation.h"
#include "backends/cuda/engine/executor/options.h"
#include "backends/cuda/engine/executor/execution.h"
#include "engine/state/kv_memory.h"
#include "engine/state/kv_offload.h"
#include "engine/models/common/session/cache.h"
#include "engine/weights/loader.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/weights/moe/kv_expert_loan.h"
#include "engine/weights/routed_cache.h"
#include "cuda_runtime_config.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/weights/memory.h"
#include "source/model_source.h"

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

template <class LoadedModel, auto Load>
class CudaEngine final : public engine::Engine {
    using Options = CudaEngineOptions;
    using Model = typename LoadedModel::Model;
    using State = typename LoadedModel::State;
    using Parts = typename LoadedModel::Parts;

public:
    using Prepared = PreparedPrompt;

    CudaEngine(
        CudaEngineOptions options,
        std::shared_ptr<const mfq::ModelSource> source)
        : options_(std::move(options)), source_(std::move(source)) {
        if (!source_) throw std::invalid_argument("CUDA engine requires a model source");
        load();
    }

    ~CudaEngine() override {
        try {
            shutdown();
        } catch (const std::exception& error) {
            std::cerr << "runtime_unload_cleanup error=" << error.what() << '\n';
        }
    }

    engine::EngineInfo info() const override { return info_; }

    engine::Admission admit(engine::EngineRequest request) override {
        return with_cuda([&] {
            return requests_.admit(
                std::move(request), text_.get(), info_, *this);
        });
    }

    void cancel(const engine::RequestId& id) override { requests_.cancel(id); }

    engine::EngineStepResult step(
        const std::vector<engine::RequestId>& eligible) override {
        return with_cuda([&] { return requests_.step(eligible, *this); });
    }

    engine::EngineStatus status() const override {
        if (!ready_) return {0, false};
        return requests_.status(exclusive());
    }

    engine::SessionResult session(
        const engine::SessionCommand& command) override {
        if (command.kind == engine::SessionCommand::Kind::memory_budget &&
            !requests_.empty())
            throw std::runtime_error("memory budgeting requires a quiescent engine");
        return with_cuda([&] {
            if (command.kind != engine::SessionCommand::Kind::memory_budget)
                return engine::control_session(*cache_, command);
            return set_budget(command);
        });
    }

    std::int64_t reload(std::int64_t context) override {
        if (context < 1) throw std::invalid_argument("reload context must be positive");
        if (!requests_.empty()) throw std::runtime_error("reload requires a quiescent engine");
        shutdown();
        options_.context_size = context;
        source_ = mfq::open_model_source(options_.model_path);
        load();
        return info_.max_context;
    }

    void shutdown() override {
        if (env_frozen() && env_flag("TYLOINFER_CUDA_ENV_DUMP", false))
            std::cerr << "cuda knobs in use:\n" << env_dump();
        ready_ = false;
        std::exception_ptr failure;
        const auto remember = [&](std::exception_ptr error) {
            if (!failure) failure = std::move(error);
        };

        if (execution_) {
            CudaScope scope(execution_->layer_placement.primary_device(), contexts_);
            remember(requests_.clear());
        } else {
            remember(requests_.clear());
        }
        text_.reset();

        if (execution_) {
            try {
                execution_->drain();
            } catch (...) {
                remember(std::current_exception());
            }
            CudaScope scope(execution_->layer_placement.primary_device(), contexts_);
            expert_space_.reset();
            clear_bindings();
            batching_.reset();
            budget_.reset();
            cache_.reset();
            graph_.reset();
            loaded_.reset();
            execution_->pipelines.clear();
            offload_.reset();
            kv_.reset();
            weights_.reset();
            execution_.reset();
        }
        contexts_.clear();
        tq_.reset();
        weight_memory_.reset();
        source_.reset();
        preparing_ = false;
        duplex_active_ = false;
        if (failure) std::rethrow_exception(failure);
    }

    engine::ControlResult control(engine::ControlRequest request) override {
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

    bool can_batch(const engine::EngineRequest& request) const {
        return batching_ && engine::batch_compatible(
            request, cache_->persistent_prefix_enabled(), bool(components().mtp));
    }

    bool exclusive() const { return duplex_active_; }
    bool mtp_available() const { return bool(components().mtp); }

    void execute(const std::vector<engine::RequestId>& eligible) {
        if (batching_) batching_->step(eligible);
    }

    mfq::StepSequence<Prepared> prepare(engine::InferenceRequest& input) {
        return components().prepare(
            language(), execution_->projector(), weights_.moe,
            execution_->collectives, execution_->layer_placement.primary_device(),
            loaded_->executor.get(), input.prompt, *input.vision);
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
            language(), state(), loaded_->executor.get(), execution_->collectives,
            execution_->forward_mode(), execution_->profiler, weights_,
            execution_->decode_graph_serial_branches,
            execution_->config.report_cuda_memory,
            execution_->dsv4_cpu_offload_layers.empty() &&
                execution_->dense_cpu_layer_count == 0 &&
                model_parallel_cuda_graph_enabled(*execution_),
            [owner = execution_.get()] { owner->drain(); }, metadata(), *graph_,
            *cache_, config_, input, output,
            batched ? nullptr : components().mtp.get(), std::move(prepared),
            batched ? batching_.get() : nullptr, id, restore_decode_workspace_);
    }

    engine::Generation generate(
        const engine::RequestId& id, engine::ExecutionRequest& request) {
        if (request.batched) graph_->invalidate();
        return engine::generate_prepared(
            *this, request.input, request.output, id, request.batched);
    }

private:
    Options options_;
    std::shared_ptr<TqExec> tq_;
    std::shared_ptr<KvMemoryManager> kv_;
    RoutedCache weights_;
    std::shared_ptr<KvOffloadStore> offload_;
    ContextSet contexts_;
    std::unique_ptr<CudaExecutionContext> execution_;
    std::shared_ptr<WeightMemoryLedger> weight_memory_;
    std::shared_ptr<const mfq::ModelSource> source_;
    CudaRuntimeConfig config_;
    std::optional<LoadedModel> loaded_;
    std::optional<DecodeGraphCache> graph_;
    std::optional<internal::TextSessionCache> cache_;
    std::optional<Budget> budget_;
    std::unique_ptr<engine::Batch> batching_;
    std::optional<ExpertSpace> expert_space_;
    std::function<void()> restore_decode_workspace_;
    std::unique_ptr<engine::TextProcessor> text_;
    engine::EngineInfo info_;
    engine::RequestExecutor requests_;
    bool ready_ = false;
    bool duplex_active_ = false;
    bool preparing_ = false;

    template <class F>
    decltype(auto) with_cuda(F&& function) {
        if (!ready_ || !execution_ || !loaded_)
            throw std::runtime_error("CUDA engine is unloaded");
        CudaScope scope(execution_->layer_placement.primary_device(), contexts_);
        mfq_tensor_backend::NoGradGuard no_grad;
        return std::forward<F>(function)();
    }

    Model& language() { return loaded_->language(); }
    State& state() { return loaded_->state(); }
    Parts& components() { return loaded_->components; }
    const Parts& components() const { return loaded_->components; }
    decltype(auto) metadata() { return (language().metadata); }

    bool online_budget_supported() const {
        return execution_->layer_placement.devices.size() <= 1 &&
            !execution_->tensor_parallel.enabled() &&
            !execution_->expert_parallel.enabled();
    }

    void restore_graph_expert_space() {
        if (weights_.tq || preparing_) return;
        prepare_moe_decode_residency(weights_.moe);
        default_context(execution_->layer_placement.primary_device())->reclaim_idle_memory();
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

            tq_ = std::make_shared<TqExec>();
            execution_ = std::make_unique<CudaExecutionContext>(
                contexts_, weights_, kv_, offload_, tq_);
            CudaScope outer_scope(
                execution_->layer_placement.primary_device(), contexts_);
            internal::setup_cuda_load(options_, *execution_);
            execution_->profiler.enabled = false;
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
                execution_->layer_placement.primary_device(), contexts_);
            weight_loader::Loader loader{
                *source_, execution_->config.weights,
                {execution_->layer_placement.primary_device(), false,
                 execution_->drop_file_cache, execution_->tensor_parallel,
                 moe_parallel_config(*execution_)},
                weights_, tq_};
            loaded_.emplace(Load(
                source_, loader, execution_->layer_placement, execution_->yarn,
                execution_->dense_cpu_layer_count, options_,
                execution_->n_gpu_layers, execution_->yarn_context_extension,
                execution_->expert_parallel.enabled(),
                execution_->config.tensor_parallel_mirror_linear_attention_scalars,
                execution_->make_executor(), execution_->collectives,
                execution_->config.gdn_transposed_state,
                execution_->kl_mmq.kv_cache_capacity,
                execution_->gemma_stage_trace, execution_->gemma_trace_layer,
                !execution_->dsv4_cpu_offload_layers.empty(),
                runtime_options::kv_vram_preallocated()));
            source_ = language().source;
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
                shutdown();
            } catch (const std::exception& error) {
                std::cerr << "runtime_load_cleanup error=" << error.what() << '\n';
            }
            std::rethrow_exception(failure);
        }
    }

    void initialize_runtime() {
        const auto& model_metadata = metadata();
        graph_.emplace(model_metadata.max_position_embeddings);
        cache_.emplace(
            config_.session_cache, config_.prefix_cache,
            internal::make_cuda_paged_prefix_cache(
                *source_, model_metadata.max_position_embeddings,
                state().supports_paged_text_session_state(), config_.prefix_cache),
            state().supports_text_session_state(),
            state().supports_text_session_state() ? 0 : 1,
            internal::cuda_has_separate_memory());
        budget_.emplace(
            *weight_memory_, weights_, *cache_,
            execution_->layer_placement.primary_device());
        if (offload_) {
            offload_->set_ram_reservation([this](std::size_t bytes) {
                cache_->reserve_ram(bytes);
                weights_.reserve(bytes);
            });
        }
        if (config_.continuous_batch.max_sequences) {
            auto batch_config = config_.continuous_batch;
            batch_config.cuda_graph &= model_parallel_cuda_graph_enabled(*execution_);
            batching_ = make_batch(
                std::shared_ptr<const Model>(loaded_->model), &state(),
                execution_->make_executor(), execution_->collectives,
                execution_->forward_mode(), weights_.moe, std::move(batch_config),
                config_.generation.prefill_chunk_size,
                runtime_options::kv_vram_preallocated(),
                execution_->dense_cpu_layer_count != 0 ||
                    !execution_->dsv4_cpu_offload_layers.empty(),
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
                    weights_, execution_->layer_placement.primary_device(),
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
        info_.max_context = model_metadata.max_position_embeddings;
        info_.vocab_size = static_cast<std::int32_t>(model_metadata.vocab_size);
        info_.model_type = model_metadata.model_type;
        components().describe(info_);
        info_.reload = true;
        text_ = engine::TextProcessor::load(
            *source_, options_.tokenizer_model, info_.vocab_size, info_.model_type);
        info_.chat = text_->chat_template_capabilities();
        info_.stop_token_ids = engine::model_stop_token_ids(
            source_->model_config_json(options_.config_path), info_.vocab_size);
        requests_ = engine::RequestExecutor(info_.max_requests);
    }

    template <class T>
    engine::ControlResult control_model(T value) {
        if constexpr (std::is_same_v<T, engine::RuntimeMetrics>) {
            return engine_detail::runtime_metrics(
                language(), components(), 3,
                expert_space_ ? &*expert_space_ : nullptr, batching_.get(),
                source_.get(), *weight_memory_, weights_, kv_.get(),
                offload_.get(), *cache_, *budget_,
                execution_->layer_placement, execution_->tensor_parallel,
                execution_->expert_parallel);
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
