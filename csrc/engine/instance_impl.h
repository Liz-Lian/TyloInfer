#pragma once

#include "engine/models/common/load.h"
#include "engine/models/common/model_config.h"
#include "env.h"
#include "engine/load.h"
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
struct Backend {
    using Options = CudaEngineOptions;
    Options options;
    std::shared_ptr<KvMemoryManager> kv;
    RoutedCache weights;
    std::shared_ptr<KvOffloadStore> offload;
    std::unique_ptr<CudaExecutionContext> execution;
    std::optional<State> state;
    ~Backend() {
        try { unload(); }
        catch (const std::exception& error) { std::cerr << "runtime_unload_cleanup error=" << error.what() << '\n'; }
    }
    template <class F> decltype(auto) visit(F&& run) {
        if (!state) throw std::runtime_error("CUDA engine is unloaded");
        return std::visit([&](auto& value) -> decltype(auto) {
            if (!value) throw std::runtime_error("CUDA engine is unloaded");
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            auto guard = value->executor->enter();
#else
            CudaScope guard(value->execution->layer_placement.primary_device(), value->execution->executor().contexts());
#endif
            mfq_tensor_backend::NoGradGuard no_grad;
            return run(*value);
        }, *state);
    }
    std::pair<EngineInfo, std::unique_ptr<TextProcessor>> load() {
        const bool dump_env = mfq::cuda::env_flag("TYLOINFER_CUDA_ENV_DUMP", false);
        mfq::cuda::env_freeze();
        if (dump_env) std::cerr << "cuda knobs at load:\n" << mfq::cuda::env_dump();
        if (!options.context_size) options.context_size = 32768;
        if (execution) throw std::logic_error("CUDA load requires retired resources");
        execution = std::make_unique<CudaExecutionContext>(weights, kv, offload);
        CudaScope scope(execution->layer_placement.primary_device(), execution->executor().contexts());
        setup_cuda_load(options, *execution);
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
            [&](auto& loaded, auto, auto, auto source) -> State {
                using LoadedModel = std::remove_cvref_t<decltype(loaded)>;
                if (offload && !loaded.components.streamed_kv())
                    throw std::invalid_argument("CUDA streamed KV requires a QSA model");
                return std::make_unique<Text<typename LoadedModel::Model, typename LoadedModel::State,
                                             typename LoadedModel::Parts>>(
                    *execution, weights, kv, offload, std::move(loaded.model),
                    std::move(loaded.owned_state), std::move(loaded.executor),
                    std::move(loaded.components), config,
                    std::move(source), weight_memory);
            });
        return visit([&](auto& state) {
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
    }
    bool exclusive() const {
        if (!state) return false;
        return std::visit([](const auto& value) { return value && value->exclusive(); }, *state);
    }
    void unload() {
        if (mfq::cuda::env_frozen() && mfq::cuda::env_flag("TYLOINFER_CUDA_ENV_DUMP", false))
            std::cerr << "cuda knobs in use:\n" << mfq::cuda::env_dump();
        std::exception_ptr failure;
        if (execution) {
            try { execution->drain(); } catch (...) { failure = std::current_exception(); }
            {
                CudaScope scope(execution->layer_placement.primary_device(), execution->executor().contexts());
                state.reset();
                execution->executor().pipelines().clear();
            }
        }
        offload.reset();
        kv.reset();
        weights.reset();
        execution.reset();
        if (failure) std::rethrow_exception(failure);
    }
};
} // namespace engine_detail

} // namespace mfq::cuda
