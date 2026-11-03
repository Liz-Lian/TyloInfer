#pragma once

#include "engine/engine.h"
#include "engine/load.h"
#include "engine/drain.h"
#include "engine/generate.h"
#include "engine/generation/generation.h"
#include "engine/generation/mtp/state.h"
#include "engine/runtime_metrics.h"
#include "engine/models/causal_model.h"
#include "engine/models/common/model_config.h"
#include "engine/models/common/session/cache.h"
#include "engine/models/prepare.h"
#include "engine/requests/processor.h"
#include "engine/scheduler/table.h"
#include "engine/weights/workspace.h"
#include "engine/weights/moe/kv_expert_loan.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/weights/memory.h"

#include <algorithm>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace mfq::cuda {

class NativeEngine;
std::unique_ptr<engine::Engine> load_native_engine(CudaEngineOptions options, NativeEngine* target = nullptr);

class NativeEngine final : public engine::Engine {
public:
    using Prepared = PreparedPrompt;

    NativeEngine(CudaEngineOptions options, CudaLoadConfig load, CudaRuntimeConfig runtime,
        std::shared_ptr<const ModelSource> source, std::shared_ptr<WeightMemoryLedger> ledger,
        std::unique_ptr<RoutedCache> weights, std::shared_ptr<KvMemoryManager> kv,
        std::shared_ptr<KvOffloadStore> offload, std::unique_ptr<Executor> root,
        std::shared_ptr<models::CausalModel> model, std::unique_ptr<models::State> state,
        std::unique_ptr<Executor> forward) {
        install(std::move(options), std::move(load), std::move(runtime), std::move(source),
            std::move(ledger), std::move(weights), std::move(kv), std::move(offload),
            std::move(root), std::move(model), std::move(state), std::move(forward));
        start();
    }
    ~NativeEngine() override {
        try { shutdown(); }
        catch (const std::exception& error) { std::cerr << "runtime_unload_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "runtime_unload_cleanup error=unknown\n"; }
    }

protected:
    engine::EngineInfo describe() const override { return info_; }
    bool ready() const override { return ready_; }
    std::unique_ptr<mfq::Executor::Scope> enter() override { return forward_->enter(); }
    std::unique_ptr<engine::ExecutionRequest> prepare_request(engine::EngineRequest request, bool batched) override {
        return engine::prepare_request(std::move(request), text_.get(), info_, batched, mtp_available());
    }
    engine::SessionResult session_command(const engine::SessionCommand& command) override {
        return with_cuda([&] {
            return command.kind == engine::SessionCommand::Kind::memory_budget
                ? set_budget(command) : engine::control_session(*cache_, command);
        });
    }
    std::int64_t reload_model(std::int64_t context) override {
        if (context < 1) throw std::invalid_argument("reload context must be positive");
        unload_model();
        options_.context_size = context;
        load_native_engine(options_, this);
        return info_.max_context;
    }
    void unload_model() override {
        if (env_frozen() && env_flag("TYLOINFER_CUDA_ENV_DUMP", false))
            std::cerr << "cuda knobs in use:\n" << env_dump();
        ready_ = false;
        std::exception_ptr failure;
        const auto remember = [&](std::exception_ptr error) { if (!failure) failure = std::move(error); };
        text_.reset();
        if (root_) {
            try { drain_cuda(*root_, *weights_, offload_.get(), load_.layer_placement, load_.tensor_parallel, load_.expert_parallel); }
            catch (...) { remember(std::current_exception()); }
            const auto release = [&] {
                expert_space_.reset();
                clear_bindings();
                budget_.reset(); cache_.reset(); graph_.reset();
                forward_.reset(); state_.reset(); model_.reset(); mtp_state_ = {};
                root_->pipelines().clear();
                offload_.reset(); kv_.reset(); weights_.reset();
            };
            try {
                CudaScope scope(load_.layer_placement.primary_device(), root_->contexts());
                release();
            } catch (...) {
                remember(std::current_exception());
                release();
            }
            root_.reset();
        }
        ledger_.reset(); source_.reset();
        if (failure) std::rethrow_exception(failure);
    }
    engine::ControlResult control_command(engine::ControlRequest request) override {
        if (!ready_) throw std::runtime_error("engine is unloaded");
        return with_cuda([&]() -> engine::ControlResult {
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
                    return engine_detail::runtime_metrics(model_->metrics(), {}, std::nullopt, false, nullptr,
                        mtp_state_, 3, expert_space_ ? &*expert_space_ : nullptr, nullptr, source_.get(),
                        *ledger_, *weights_, kv_.get(), offload_.get(), *cache_, *budget_,
                        load_.layer_placement, load_.tensor_parallel, load_.expert_parallel);
                } else throw std::invalid_argument("model has no duplex component");
            }, std::move(request));
        });
    }

public:
    bool can_batch(const engine::EngineRequest&) const override { return false; }
    bool exclusive() const override { return false; }
    bool mtp_available() const { return false; }
    void execute(const std::vector<engine::RequestId>&) override {}
    mfq::StepSequence<Prepared> prepare(engine::InferenceRequest& input) {
        return model_->prepare(input.prompt, *input.vision, *state_);
    }
    auto advance_preparation(mfq::StepSequence<Prepared>& preparation) {
        internal::PrefillCudaTimer timer;
        auto step = preparation.next();
        TYLOINFER_CUDA_CHECK(cudaEventRecord(timer.finished_event(), mfq_get_current_cuda_stream()));
        return std::pair{std::move(step), timer.elapsed_ms()};
    }
    engine::Generation generate_text(engine::InferenceRequest& input, engine::InferenceOutput& output,
        std::optional<Prepared> prepared, const engine::RequestId& id, bool) {
        return internal::generate(*model_, *state_, forward_.get(), root_->collectives(),
            root_->forward_mode(load_.layer_placement.primary_device()), root_->profiler(), *weights_,
            root_->mode().ffn.serial, load_.report_cuda_memory,
            load_.dsv4_cpu_offload_layers.empty() && load_.dense_cpu_layer_count == 0 &&
                model_parallel_cuda_graph_enabled(load_, *root_, *weights_),
            [this] { drain_cuda(*root_, *weights_, offload_.get(), load_.layer_placement, load_.tensor_parallel, load_.expert_parallel); },
            model_info_, source_.get(), *graph_, *cache_, runtime_, input, output, mtp_state_,
            static_cast<MtpModule*>(nullptr), std::move(prepared), nullptr, id, restore_);
    }
    engine::Generation generate(const engine::RequestId& id, engine::ExecutionRequest& request) override {
        return engine::generate_prepared(*this, request.input, request.output, id, request.batched);
    }

private:
    friend std::unique_ptr<engine::Engine> load_native_engine(CudaEngineOptions, NativeEngine*);
    CudaEngineOptions options_;
    CudaLoadConfig load_;
    CudaRuntimeConfig runtime_;
    std::shared_ptr<const ModelSource> source_;
    std::shared_ptr<WeightMemoryLedger> ledger_;
    std::unique_ptr<RoutedCache> weights_;
    std::shared_ptr<KvMemoryManager> kv_;
    std::shared_ptr<KvOffloadStore> offload_;
    std::unique_ptr<Executor> root_;
    std::shared_ptr<models::CausalModel> model_;
    std::unique_ptr<models::State> state_;
    std::unique_ptr<Executor> forward_;
    models::Info model_info_{};
    engine::mtp::State mtp_state_;
    std::optional<DecodeGraphCache> graph_;
    std::optional<internal::TextSessionCache> cache_;
    std::optional<Budget> budget_;
    std::optional<ExpertSpace> expert_space_;
    std::function<void()> restore_;
    std::unique_ptr<engine::TextProcessor> text_;
    engine::EngineInfo info_;
    bool ready_ = false;

    template <class F> decltype(auto) with_cuda(F&& work) {
        if (!ready_) throw std::runtime_error("CUDA engine is unloaded");
        auto scope = forward_->enter();
        mfq_tensor_backend::NoGradGuard no_grad;
        return std::forward<F>(work)();
    }
    void restore_graph_expert_space() {
        if (weights_->tq) return;
        prepare_moe_decode_residency(weights_->moe);
        default_context(load_.layer_placement.primary_device())->reclaim_idle_memory();
        if (expert_space_) expert_space_->restore(runtime_options::workspace_reserve_bytes(), source_.get());
    }
    void clear_bindings() noexcept {
        restore_ = {};
        if (kv_) kv_->set_reservation({});
        set_moe_workspace_reservation(weights_->moe, {});
        if (offload_) offload_->set_ram_reservation({});
    }
    engine::SessionResult set_budget(const engine::SessionCommand& command) {
        if (load_.layer_placement.devices.size() > 1 || load_.tensor_parallel.enabled() || load_.expert_parallel.enabled())
            throw std::invalid_argument("online memory budgeting requires a single-device CUDA worker");
        return budget_->set(command, [this] {
            if (expert_space_) expert_space_->release();
            finish_moe_expert_exchanges(weights_->moe);
            graph_->invalidate();
            state_->reset(1);
        });
    }
    void install(CudaEngineOptions options, CudaLoadConfig load, CudaRuntimeConfig runtime,
        std::shared_ptr<const ModelSource> source, std::shared_ptr<WeightMemoryLedger> ledger,
        std::unique_ptr<RoutedCache> weights, std::shared_ptr<KvMemoryManager> kv,
        std::shared_ptr<KvOffloadStore> offload, std::unique_ptr<Executor> root,
        std::shared_ptr<models::CausalModel> model, std::unique_ptr<models::State> state,
        std::unique_ptr<Executor> forward) {
        options_ = std::move(options); load_ = std::move(load); runtime_ = std::move(runtime);
        source_ = std::move(source); ledger_ = std::move(ledger); weights_ = std::move(weights);
        kv_ = std::move(kv); offload_ = std::move(offload); root_ = std::move(root);
        model_ = std::move(model); state_ = std::move(state); forward_ = std::move(forward);
        try {
            {
                CudaScope scope(load_.layer_placement.primary_device(), root_->contexts());
                WeightMemoryLoadScope memory_load(*ledger_);
                mfq_tensor_backend::NoGradGuard no_grad;
                model_info_ = model_->info();
                graph_.emplace(model_info_.max_context);
                cache_.emplace(runtime_.session_cache, runtime_.prefix_cache,
                    internal::make_cuda_paged_prefix_cache(*source_, model_info_.max_context,
                        state_->supports_paged_text_session_state(), runtime_.prefix_cache),
                    state_->supports_text_session_state(), state_->supports_text_session_state() ? 0 : 1,
                    internal::cuda_has_separate_memory());
                budget_.emplace(*ledger_, *weights_, *cache_, load_.layer_placement.primary_device());
                if (offload_) offload_->set_ram_reservation([this](std::size_t bytes) { cache_->reserve_ram(bytes); weights_->reserve(bytes); });
                if (runtime_.continuous_batch.max_sequences)
                    throw std::invalid_argument("continuous batching is unavailable for this model");
                if (weights_->moe) {
                    auto loan = kv_ ? kv_->expert_loan() : nullptr;
                    if (!loan && moe_expert_cache_ssd_experts(weights_->moe)) {
                        loan = std::make_shared<KvExpertLoan>();
                        set_moe_kv_expert_loan(weights_->moe, loan);
                    }
                    if (loan) expert_space_.emplace(*weights_, load_.layer_placement.primary_device(), std::move(loan));
                    restore_ = [this] { restore_graph_expert_space(); };
                    set_moe_workspace_reservation(weights_->moe,
                        [this](std::size_t bytes) { budget_->reclaim(bytes, expert_space_ ? &*expert_space_ : nullptr); },
                        [this](std::size_t bytes) { cache_->reserve_ram(bytes); weights_->reserve(bytes); });
                }
                if (const auto bytes = runtime_options::resident_budget_bytes()) {
                    engine::SessionCommand command; command.kind = engine::SessionCommand::Kind::memory_budget; command.bytes = bytes;
                    set_budget(command);
                }
                restore_graph_expert_space();
                info_ = {};
                info_.max_requests = std::max<std::size_t>(1, runtime_.continuous_batch.max_sequences);
                info_.max_context = model_info_.max_context;
                info_.vocab_size = static_cast<std::int32_t>(model_info_.vocab_size);
                info_.model_type = model_info_.model_type;
                info_.capabilities = model_->capabilities();
                const auto& caps = info_.capabilities;
                info_.duplex = caps.full_duplex;
                info_.multimodal = caps.image_input || caps.video_input || caps.audio_input || caps.audio_output || caps.full_duplex;
                info_.reload = true;
                text_ = engine::TextProcessor::load(*source_, options_.tokenizer_model, info_.vocab_size, info_.model_type);
                if (text_) info_.chat = text_->chat_template_capabilities();
                info_.stop_token_ids = models::stop_token_ids(models::config_json(*source_, options_.config_path), info_.vocab_size);
                ready_ = true;
            }
        } catch (...) {
            const auto failure = std::current_exception();
            try { unload_model(); }
            catch (const std::exception& error) { std::cerr << "runtime_load_cleanup error=" << error.what() << '\n'; }
            catch (...) { std::cerr << "runtime_load_cleanup error=unknown\n"; }
            std::rethrow_exception(failure);
        }
    }
};

} // namespace mfq::cuda
