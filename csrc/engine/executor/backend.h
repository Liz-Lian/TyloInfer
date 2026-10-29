#pragma once

#include "backends/cuda/engine/models/load.h"
#include "env.h"
#include "cuda_runtime_config.h"
#include "cuda_model_plan.h"
#include "backends/cuda/engine/executor/execution.h"
#include "engine/executor/tq.h"
#include "engine/executor/text.h"
#include "backends/cuda/engine/state/kv_memory.h"
#include "backends/cuda/engine/state/kv_offload.h"
#include "backends/cuda/engine/executor/options.h"
#include "backends/cuda/engine/state/session/text_cache.h"
#include "engine/executor/engine.h"
#include "engine/requests/processor.h"
#include "weight_memory.h"

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
    std::shared_ptr<TqExec> tq;
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
            CudaScope guard(*value->execution);
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
        tq = std::make_shared<TqExec>();
        execution = std::make_unique<CudaExecutionContext>(weights, kv, offload, tq);
        CudaScope scope(*execution);
        setup_cuda_load(options, *execution);
        execution->profiler.enabled = false;
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
            [&](auto& loaded, auto, auto) -> State {
                using LoadedModel = std::remove_cvref_t<decltype(loaded)>;
                using Model = typename LoadedModel::Model;
                if constexpr (!std::is_same_v<Model, Qwen4CausalLm>)
                    if (offload) throw std::invalid_argument("CUDA streamed KV requires a QSA model");
                auto source = loaded.language().source;
                return std::make_unique<Text<LoadedModel>>(
                    *execution, weights, kv, offload, std::move(loaded), config,
                    std::move(source), weight_memory);
            });
        return visit([&](auto& state) {
            EngineInfo info;
            info.max_requests = std::max<std::size_t>(1, config.continuous_batch.max_sequences);
            info.max_context = state.metadata.max_position_embeddings;
            info.vocab_size = static_cast<int32_t>(state.metadata.vocab_size);
            info.model_type = state.metadata.model_type;
            const auto component_state = state.components.state();
            const bool composite = [&] {
                if constexpr (requires { state.components.composite; })
                    return bool(state.components.composite);
                else
                    return false;
            }();
            info.multimodal = component_state.vision_available || composite;
            info.reload = true;
            const auto caps = cuda_runtime_capabilities(state.components.graph, state.components.plan,
                component_state, info.multimodal, composite);
            info.duplex = caps.full_duplex;
            info.capabilities = {state.components.graph.architecture, caps.text,
                caps.image_input, caps.video_input, caps.audio_input, caps.audio_output,
                caps.full_duplex, caps.mtp, "model-graph+cuda-adapters"};
            auto text = TextProcessor::load(*state.source, options.tokenizer_model,
                info.vocab_size, info.model_type);
            info.stop_token_ids = model_stop_token_ids(
                load_model_config_json(*state.source, options.config_path), info.vocab_size);
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
                CudaScope scope(*execution);
                state.reset();
                execution->pipelines.clear();
            }
        }
        offload.reset();
        kv.reset();
        weights.reset();
        execution.reset();
        tq.reset();
        if (failure) std::rethrow_exception(failure);
    }
};
} // namespace engine_detail

} // namespace mfq::cuda
