#pragma once

#include "env.h"
#include "engine/backends/cuda/executor/config.h"
#include "engine/drain.h"
#include "engine/models/common/components.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/models/qwen35/batch.h"
#include "engine/models/qwen4_exp/components.h"
#include "engine/models/qwen4_exp/batch.h"
#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/nemotron_h/components.h"
#endif
#include "engine/models/common/mtp.h"
#include "engine/models/common/prompt.h"
#include "engine/models/metadata.h"
#include "engine/models/causal_model.h"
#include "engine/backends/cuda/executor/execution.h"
#include "engine/budget.h"
#include "engine/backends/cuda/executor/tq.h"
#include "engine/generation/cuda.h"
#include "engine/weights/loader.h"
#include "engine/state/kv_offload.h"
#include "engine/state/kv_memory.h"
#include "engine/state/media.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/generation/batch.h"
#include "engine/models/session/cache.h"
#include "engine/resource_metrics.h"
#include "engine/generation/generation.h"
#include "engine/generation/prefill_budget.h"
#include "engine/generation/mtp/metrics.h"
#include "engine/generation/mtp/state.h"
#include "engine/models/source.h"
#include "engine/weights/memory.h"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace tyloi::cuda::engine_detail {

using namespace tyloi::engine;
using namespace tyloi::cuda::internal;
using tyloi::cuda::make_batch;

template <class M, class S>
std::unique_ptr<engine::Batch> make_batch(
    std::shared_ptr<const M>, S*, std::unique_ptr<Executor>, Collectives&,
    Executor::Mode, tyloi::engine::Budget&, tyloi::engine::ContinuousBatchConfig,
    std::int64_t, bool, bool, int) {
    throw std::invalid_argument("continuous batching is unavailable for this model");
}

template <class M, class S, class P>
struct Text {
    using Model = M;
    using State = S;
    using Parts = P;

    std::shared_ptr<tyloi::weights::Memory> weight_memory;
    CudaExecutionContext* execution;
    Executor& root_executor;
    ::tyloi::weights::Cache& weights;
    std::shared_ptr<KvMemoryManager>& kv;
    std::shared_ptr<KvOffloadStore>& offload;
    tyloi::engine::Config config;
    std::shared_ptr<const tyloi::ModelSource> source;
    std::shared_ptr<Model> model;
    std::unique_ptr<State> owned_state;
    std::unique_ptr<Executor> executor;
    tyloi::cache::Media images;
    Parts components;
    Model& language;
    State& state;
    const tyloi::models::Info info;
    DecodeGraphCache graph;
    tyloi::engine::SessionCache<CudaSessionOps> cache;
    std::optional<tyloi::engine::Budget> budget;
    tyloi::engine::mtp::State mtp_state;
    std::unique_ptr<tyloi::engine::Batch> batching;
    bool duplex_active = false;
    tyloi::models::MtpState& predictor_state() {
        if constexpr (requires { state.predictor; }) {
            if (!state.predictor)
                throw std::logic_error("MTP predictor state is unavailable");
            return *state.predictor;
        } else {
            throw std::logic_error("model state has no MTP predictor owner");
        }
    }

    void release_request_resources() noexcept {
#ifndef TYLOINFER_NATIVE_CUDA
        if constexpr (std::is_same_v<Model, qwen4_exp::Model>) {
            if (component_predictor(language, components)) {
                try {
                    qwen4_release_decode_graphs(language, state);
                } catch (...) {
                }
            }
        }
#endif
        try {
            graph.invalidate();
        } catch (...) {
        }
        owned_state.reset();
        components = Parts{};
    }

    bool online_budget_supported() const {
        return execution->config.layer_placement.devices.size()<=1 && !execution->config.tensor_parallel.enabled()
            && !execution->config.expert_parallel.enabled();
    }

    Text(CudaExecutionContext& borrowed, ::tyloi::weights::Cache& borrowed_weights,
            std::shared_ptr<KvMemoryManager>& borrowed_kv,
            std::shared_ptr<KvOffloadStore>& borrowed_offload,
            std::shared_ptr<Model> model_owner, std::unique_ptr<State> state_owner,
            std::unique_ptr<Executor> executor_owner, Parts model_components,
            tyloi::engine::Config runtime_config,
            std::shared_ptr<const tyloi::ModelSource> model_source,
            std::shared_ptr<tyloi::weights::Memory> ledger)
        : weight_memory(std::move(ledger)), execution(&borrowed), root_executor(borrowed.executor()),
          weights(borrowed_weights),
          kv(borrowed_kv), offload(borrowed_offload), config(std::move(runtime_config)),
          source(std::move(model_source)),
          model(std::move(model_owner)), owned_state(std::move(state_owner)),
          executor(std::move(executor_owner)), components(std::move(model_components)),
          language(*model), state([this]() -> State& {
              if constexpr (std::is_same_v<Model, State>) return language;
              else return *owned_state;
          }()),
          info(tyloi::models::model_info(language)),
          graph(info.max_context),
          cache(config.session_cache, config.prefix_cache,
              make_cuda_paged_prefix_cache(*source, info.max_context,
                  state.supports_paged_text_session_state(), config.prefix_cache),
                state.supports_text_session_state(), state.supports_text_session_state() ? 0 : 1,
                cuda_has_separate_memory()) {
        try {
#ifdef TYLOINFER_NATIVE_CUDA
        root_executor.configure_kv(kv, offload);
        executor->configure_kv(kv, offload);
#endif
        budget.emplace(*weight_memory, weights, cache, root_executor);
        if (offload) offload->bind_budget(&*budget);
        if (config.continuous_batch.max_sequences) {
#ifndef TYLOINFER_NATIVE_CUDA
            if constexpr (std::is_same_v<Model,qwen4_exp::Model>) {
                batching = make_batch(language, *execution, *budget, config.continuous_batch,
                    config.generation.prefill_chunk_size);
            }
            else
#endif
            {
                auto batch_config = config.continuous_batch;
                batch_config.cuda_graph &= model_parallel_cuda_graph_enabled(execution->config, execution->executor(), execution->weights);
                batching = make_batch(
                    std::shared_ptr<const Model>(model), &state, execution->executor().fork(model_parallel_primary_device(execution->config)),
                    execution->executor().collectives(), execution->executor().forward_mode(execution->config.layer_placement.primary_device()),
                    *budget, std::move(batch_config),
                    config.generation.prefill_chunk_size,
                    ::tyloi::cuda::options::kv_vram_preallocated(),
                    execution->config.dense_cpu_layer_count != 0 ||
                        !execution->config.dsv4_cpu_offload_layers.empty(),
                    execution->config.weights.min_slots);
            }
        }
#ifdef TYLOINFER_NATIVE_CUDA
        budget->bind(component_predictor(language, components) != nullptr);
#endif
        if(const auto bytes=::tyloi::cuda::options::resident_budget_bytes()) {
            SessionCommand command;
            command.kind=SessionCommand::Kind::memory_budget;
            command.bytes=bytes;
            session(command);
        }
#ifndef TYLOINFER_NATIVE_CUDA
        if constexpr(std::is_same_v<Model,qwen4_exp::Model>) {
            qwen4_warmup_decode_graphs(language,state);
        }
#endif
        weights.prepare(::tyloi::weights::Cache::Stage::loaded,
            tyloi::Device(tyloi::kCUDA, execution->config.layer_placement.primary_device()), source.get());
        } catch (...) {
            const auto failure = std::current_exception();
            try {
                drain_cuda(root_executor, weights, offload.get(), execution->config.layer_placement,
                           execution->config.tensor_parallel, execution->config.expert_parallel);
            }
            catch (const std::exception& error) { std::cerr << "runtime_load_cleanup error=" << error.what() << '\n'; }
            batching.reset();
            release_request_resources();
            clear_bindings();
            std::rethrow_exception(failure);
        }
    }
    bool can_batch(const EngineRequest& request) const {
#ifndef TYLOINFER_NATIVE_CUDA
        if constexpr (std::is_same_v<Model,qwen4_exp::Model>)
            return batching && !request.input.media && request.input.cache_plan.session_id.empty() &&
                !(request.input.sampling.enable_mtp && component_predictor(language, components));
#endif
        return batching && batch_compatible(request, cache.persistent_prefix_enabled(), mtp_available());
    }
    void clear_bindings() noexcept {
        if(kv)kv->bind_budget(nullptr);
        if (budget) budget->unbind();
        if (offload) offload->bind_budget(nullptr);
    }
    ~Text() {
        clear_bindings();
        batching.reset();
        release_request_resources();
    }
    SessionResult session(const SessionCommand& command) {
        if(command.kind!=SessionCommand::Kind::memory_budget)return cache.control(command);
        if(!online_budget_supported())throw std::invalid_argument("online memory budgeting requires a single-device CUDA worker");
        auto change = budget->change(command);
        weights.settle();
        graph.invalidate();
#ifndef TYLOINFER_NATIVE_CUDA
        if constexpr(std::is_same_v<Model,qwen4_exp::Model>) {
            qwen4_release_decode_graphs(language,state);
        }
#endif
        if(batching)batching->release_idle_buffers();
        if(auto* mtp=component_predictor(language, components)) {
#ifdef TYLOINFER_NATIVE_CUDA
            for (const auto& [device, bytes] : mtp->reset_growth(predictor_state(), 1))
                if (bytes) budget->reserve_kv(device, tyloi::engine::prefill_budget_add(bytes, std::size_t(64) << 20));
#endif
            mtp->reset(predictor_state(), 1);
        }
#ifdef TYLOINFER_NATIVE_CUDA
        if constexpr (std::derived_from<State, tyloi::models::State>)
            for (const auto& [device, bytes] : state.reset_growth(1))
                if (bytes) budget->reserve_kv(device, tyloi::engine::prefill_budget_add(bytes, std::size_t(64) << 20));
#endif
        state.reset(1);
        return change.apply();
    }
    bool exclusive() const { return duplex_active; }
    bool mtp_available() const { return component_predictor(language, components) != nullptr; }
    void execute(const std::vector<RequestId>& eligible) {
        if (batching) batching->step(eligible);
    }
    using Prepared = tyloi::models::Prepared;
    tyloi::StepSequence<Prepared> prepare(InferenceRequest& input, State& state);
    Generation generate_text(InferenceRequest& input, InferenceOutput& output, State& state,
                             std::optional<Prepared> prepared, const RequestId& id, bool batched);
    Generation generate(const RequestId& id, InferenceRequest& input, InferenceOutput& output, bool batched);
    template <class T> ControlResult control(T value) {
        if constexpr (std::is_same_v<T, GetMetrics>) {
            const auto& model = std::as_const(language);
            const auto& parts = std::as_const(components);
            const auto component_values = tyloi::cuda::component_metrics(model, parts);
            Metrics profile_metrics;
            if constexpr (requires { parts.profile_metrics(model,state); })
                profile_metrics = parts.profile_metrics(model,state);
            std::optional<Metrics> memory;
#ifdef TYLOINFER_NATIVE_CUDA
            if constexpr (std::derived_from<State, tyloi::models::State>) memory = state.metrics();
#else
            if constexpr (requires { parts.memory_metrics(model,state); })
                memory = parts.memory_metrics(model,state);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
            const bool streamed_kv = info.streamed_kv;
#else
            const bool streamed_kv = [] {
                if constexpr (requires { Parts::streamed_kv(); }) return Parts::streamed_kv();
                else return false;
            }();
#endif
            const auto* mtp = component_predictor(model, parts);
            const auto batch_metrics = batching ? batching->metrics() : Metrics{};
#ifdef TYLOINFER_NATIVE_CUDA
            auto execution_metrics = root_executor.metrics();
#else
            auto execution_metrics = kv ? kv->metrics() : Metrics{};
            if (offload) {
                const auto values = offload->metrics();
                execution_metrics.insert(execution_metrics.end(), values.begin(), values.end());
            }
#endif
            Metrics predictor_metrics;
            if (mtp) {
                tyloi::engine::mtp::append_generation_metrics(predictor_metrics, mtp_state.stats);
            }
            return tyloi::engine::resource_metrics(component_values,profile_metrics,std::move(memory),streamed_kv,
                batch_metrics,execution_metrics,predictor_metrics,source.get(),
                *weight_memory,weights,cache,*budget,
                execution->executor().memory(cuda_devices(execution->config)),
                tyloi::Device(tyloi::kCUDA, execution->config.layer_placement.primary_device()),
                execution->config.layer_placement.devices.size() <= 1 &&
                    !execution->config.tensor_parallel.enabled() && !execution->config.expert_parallel.enabled());
        } else {
            if constexpr (std::is_same_v<Parts, std::unique_ptr<tyloi::cuda::minicpmo45::Components>>) {
                if (!components) throw std::invalid_argument("model has no duplex component");
                ControlResult result;
                if constexpr (std::is_same_v<T, MfqDuplexSessionParams>) {
                    components->start(value);
                    result = std::monostate{};
                } else if constexpr (std::is_same_v<T, MfqDuplexStepInput>) {
                    result = components->step(value);
                } else {
                    components->stop();
                    result = std::monostate{};
                }
                if constexpr (std::is_same_v<T, MfqDuplexSessionParams>) duplex_active = true;
                else if constexpr (!std::is_same_v<T, MfqDuplexStepInput>) duplex_active = false;
                return result;
            } else {
                throw std::invalid_argument("model has no duplex component");
            }
        }
    }
};

} // namespace tyloi::cuda::engine_detail
