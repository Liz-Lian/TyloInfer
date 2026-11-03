#pragma once

#include "engine/models/common/load.h"
#include "engine/models/common/model_config.h"
#include "env.h"
#include "engine/load.h"
#include "engine/drain.h"
#include "engine/models/common/capabilities.h"
#include "engine/backends/cuda/native/scope.h"
#include "engine/backends/cuda/executor/execution.h"
#include "engine/models/common/text.h"
#include "engine/state/kv_memory.h"
#include "engine/state/kv_offload.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/models/common/session/cache.h"
#include "engine/engine.h"
#include "engine/requests/processor.h"
#include "engine/scheduler/table.h"
#include "engine/models/prepare.h"
#include "engine/weights/memory.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace mfq::cuda {
namespace engine_detail {
using namespace mfq::engine;
using namespace mfq::cuda::internal;

template <class State, auto Load>
class EngineInstance final : public Engine {
    template <class F> decltype(auto) visit(F&& run) {
        if (!state) throw std::runtime_error("CUDA engine is unloaded");
        return std::visit([&](auto& value) -> decltype(auto) {
            if (!value) throw std::runtime_error("CUDA engine is unloaded");
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            auto guard = value->executor->enter();
#else
            CudaScope guard(value->execution->config.layer_placement.primary_device(), value->execution->executor().contexts());
#endif
            mfq_tensor_backend::NoGradGuard no_grad;
            return run(*value);
        }, *state);
    }

  public:
    explicit EngineInstance(CudaEngineOptions options) : options{std::move(options)} {
        load();
        start();
    }
    ~EngineInstance() override {
        try { shutdown(); }
        catch (const std::exception& error) { std::cerr << "runtime_unload_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "runtime_unload_cleanup error=unknown\n"; }
    }
  protected:
    EngineInfo describe() const override { return info_; }
    bool ready() const override { return loaded_; }
    bool can_batch(const EngineRequest& request) const override {
        return std::visit([&](const auto& ops) { return ops->can_batch(request); }, *state);
    }
    std::unique_ptr<mfq::Executor::Scope> enter() override {
        return std::visit([](auto& ops) -> std::unique_ptr<mfq::Executor::Scope> {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            return ops->executor->enter();
#else
            return ops->executor->enter(ops->execution->config.layer_placement.primary_device());
#endif
        }, *state);
    }
    void prepare_request(Request& request, bool batched) override {
        const bool mtp = std::visit([](const auto& ops) { return ops->mtp_available(); }, *state);
        engine::prepare_request(request, text_.get(), info_, batched, mtp);
    }
    Generation generate(const RequestId& id, Request& request) override {
        return std::visit([&](auto& ops) { return ops->generate(id, request.input, *request.output, request.batched); }, *state);
    }
    void execute(const std::vector<RequestId>& eligible) override {
        std::visit([&](auto& ops) { ops->execute(eligible); }, *state);
    }
    SessionResult session_command(const SessionCommand& command) override {
        return visit([&](auto& ops) {
            if constexpr(requires { ops.session(command); })return ops.session(command);
            else return control_session(ops.cache, command);
        });
    }
    ControlResult control_command(ControlRequest request) override {
        if (!loaded_) throw std::runtime_error("engine is unloaded");
        return std::visit([&](auto&& value) -> ControlResult {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, DecodeTokens> || std::is_same_v<T, CountTokens> ||
                          std::is_same_v<T, PrepareDuplex> || std::is_same_v<T, PrepareDuplexStep>) {
                if (!text_) throw std::invalid_argument("text control requires a tokenizer");
                if constexpr (std::is_same_v<T, DecodeTokens>)
                    return text_->decode_tokens(value.tokens, value.excluded);
                else if constexpr (std::is_same_v<T, CountTokens>)
                    return TokenCount{text_->count_tokens(value.prompt), info_.max_context};
                else if constexpr (std::is_same_v<T, PrepareDuplex>) {
                    text_->prepare_duplex_session(value.prompt, value.parameters);
                    return std::move(value.parameters);
                } else {
                    text_->prepare_duplex_step(value.text, value.input);
                    return std::move(value.input);
                }
            } else
                return visit([&](auto& ops) { return ops.control(std::move(value)); });
        }, std::move(request));
    }
    void unload_model() override {
        loaded_ = false;
        text_.reset();
        unload();
    }

  private:
    friend void ::mfq::cuda::reload_cuda_engine(engine::Engine&, CudaEngineOptions);
    CudaEngineOptions options;
    std::shared_ptr<KvMemoryManager> kv;
    RoutedCache weights;
    std::shared_ptr<KvOffloadStore> offload;
    std::unique_ptr<Executor> executor;
    std::unique_ptr<CudaExecutionContext> execution;
    std::optional<State> state;
    std::unique_ptr<TextProcessor> text_;
    EngineInfo info_;
    bool loaded_ = false;
    void load() {
        try {
            EngineInfo info;
            std::unique_ptr<TextProcessor> text;
            {
                const bool dump_env = mfq::cuda::env_flag("TYLOINFER_CUDA_ENV_DUMP", false);
                mfq::cuda::env_freeze();
                if (dump_env) std::cerr << "cuda knobs at load:\n" << mfq::cuda::env_dump();
                if (!options.context_size) options.context_size = 32768;
                if (execution) throw std::logic_error("CUDA load requires retired resources");
                auto [execution_config, numeric_options] = mfq::cuda::load_cuda_execution_config();
                executor = std::make_unique<Executor>(std::move(numeric_options), 0);
                execution = std::make_unique<CudaExecutionContext>(
                    *executor, std::move(execution_config), weights, kv, offload);
                CudaScope scope(execution->config.layer_placement.primary_device(), execution->executor().contexts());
                setup_cuda_load(options, execution->config, *executor, weights, kv, offload);
                execution->executor().profiler().enabled = false;
                auto weight_memory=std::make_shared<WeightMemoryLedger>();
                const WeightMemoryLoadScope memory_load(*weight_memory);
                mfq_tensor_backend::NoGradGuard no_grad;
                auto config = resolve_cuda_runtime_config(options);
                const auto kv_options = runtime_options::qsa_kv_offload();
                kv=std::make_shared<KvMemoryManager>();
                if (kv_options.gpu_budget_bytes) {
                    kv.reset();
                    KvOffloadConfig kv;
                    kv.gpu_budget_bytes = kv_options.gpu_budget_bytes;
                    kv.ram_budget_bytes = cuda_has_separate_memory() ? kv_options.ram_budget_bytes : 0;
                    kv.buffer_bytes = std::min<std::size_t>(kv.buffer_bytes, kv.gpu_budget_bytes / 4);
                    kv.directory = kv_options.directory;
                    offload = std::make_shared<KvOffloadStore>(std::move(kv));
                }
                state = Load(*execution, options, true,
                    [&](auto& model, auto& model_state, auto& model_executor, auto& components,
                        auto, auto, auto source) -> State {
                        using Model = typename std::remove_reference_t<decltype(model)>::element_type;
                        using ModelState = typename std::remove_reference_t<decltype(model_state)>::element_type;
                        using Parts = std::remove_cvref_t<decltype(components)>;
                        if (offload && !components.streamed_kv())
                            throw std::invalid_argument("CUDA streamed KV requires a QSA model");
                        return std::make_unique<Text<Model, ModelState, Parts>>(
                            *execution, weights, kv, offload, std::move(model),
                            std::move(model_state), std::move(model_executor),
                            std::move(components), config,
                            std::move(source), weight_memory);
                    });
                auto result = visit([&](auto& state) {
                    EngineInfo info;
                    info.max_requests = std::max<std::size_t>(1, config.continuous_batch.max_sequences);
                    info.max_context = state.info.max_context;
                    info.vocab_size = static_cast<int32_t>(state.info.vocab_size);
                    info.model_type = state.info.model_type;
                    describe_model(state.language, state.components, info);
                    info.reload = true;
                    auto text = TextProcessor::load(*state.source, options.tokenizer_model,
                        info.vocab_size, info.model_type);
                    info.stop_token_ids = mfq::models::stop_token_ids(
                        mfq::models::config_json(*state.source, options.config_path), info.vocab_size);
                    return std::pair{std::move(info), std::move(text)};
                });
                info = std::move(result.first);
                text = std::move(result.second);
            }
            info_ = std::move(info);
            text_ = std::move(text);
            if (text_) info_.chat = text_->chat_template_capabilities();
            loaded_ = true;
        } catch (...) {
            unload_model();
            throw;
        }
    }
    bool exclusive() const override {
        if (!state) return false;
        return std::visit([](const auto& value) { return value && value->exclusive(); }, *state);
    }
    void unload() {
        if (mfq::cuda::env_frozen() && mfq::cuda::env_flag("TYLOINFER_CUDA_ENV_DUMP", false))
            std::cerr << "cuda knobs in use:\n" << mfq::cuda::env_dump();
        std::exception_ptr failure;
        if (execution) {
            try {
                drain_cuda(*executor, weights, offload.get(), execution->config.layer_placement,
                           execution->config.tensor_parallel, execution->config.expert_parallel);
            } catch (...) { failure = std::current_exception(); }
            {
                CudaScope scope(execution->config.layer_placement.primary_device(), execution->executor().contexts());
                state.reset();
                execution->executor().pipelines().clear();
            }
        }
        offload.reset();
        kv.reset();
        weights.reset();
        execution.reset();
        executor.reset();
        if (failure) std::rethrow_exception(failure);
    }
};
} // namespace engine_detail

} // namespace mfq::cuda
