#pragma once

#include "env.h"
#include "engine/load.h"
#include "engine/drain.h"
#include "engine/models/common/components.h"
#include "engine/models/common/mtp.h"
#include "engine/models/common/prompt.h"
#include "engine/models/common/causal_metadata.h"
#include "engine/models/causal_model.h"
#include "engine/weights/cache_internal.h"
#include "engine/backends/cuda/executor/execution.h"
#include "engine/budget.h"
#include "engine/backends/cuda/executor/tq.h"
#include "engine/generate.h"
#include "engine/weights/loader.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/state/kv_offload.h"
#include "engine/state/kv_memory.h"
#include "engine/weights/moe/kv_expert_loan.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/generation/batch.h"
#include "engine/models/common/session/cache.h"
#include "engine/runtime_metrics.h"
#include "engine/generation/generation.h"
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

template <class M, class S>
std::unique_ptr<engine::Batch> make_batch(
    std::shared_ptr<const M>, S*, std::unique_ptr<Executor>, Collectives&,
    Executor::Mode, std::shared_ptr<MoeExpertCache>, tyloi::engine::ContinuousBatchConfig,
    std::int64_t, bool, bool, int) {
    throw std::invalid_argument("continuous batching is unavailable for this model");
}

template <class M, class S = M, class P = RuntimeComponents<M>>
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
    tyloi::engine::RuntimeConfig config;
    std::shared_ptr<const tyloi::ModelSource> source;
    std::optional<tyloi::engine::Budget> budget;
    std::shared_ptr<Model> model;
    std::unique_ptr<State> owned_state;
    std::unique_ptr<Executor> executor;
    MediaCache images;
    Parts components;
    Model& language;
    State& state;
    const tyloi::models::Info info;
    DecodeGraphCache graph;
    tyloi::engine::SessionCache<CudaSessionOps> cache;
    tyloi::engine::mtp::State mtp_state;
    std::unique_ptr<tyloi::engine::Batch> batching;
    bool duplex_active = false;
    int graph_draft_depth=3;
    bool preparing_graph_depth=true;

    void configure_decode_budget() {
        std::size_t spare = 0;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if constexpr(std::is_same_v<Model,qwen4_exp::Model>) {
            const auto& cache = ::tyloi::weights::detail::cache(weights);
            if (!cache.tq && cache.moe && components.mtp) {
                const auto maximum=qwen4_mtp_graph_reserve(language,components.mtp.get());
                const auto active=qwen4_mtp_graph_reserve(language,components.mtp.get(),graph_draft_depth);
                if(active<maximum)spare=maximum-active;
            }
        }
#endif
        budget->configure_decode(spare,graph_draft_depth);
    }

    void prepare_graph_depth(const InferenceRequest& input) {
        if constexpr(std::is_same_v<Model,qwen4_exp::Model>) {
            if(!components.mtp)return;
            const auto* warmup=tyloi::cuda::env_lookup("TYLOINFER_QWEN4_WARMUP_GRAPHS");
            if(!language.reuse_decode_graphs || !execution->config.moe_pipeline || (warmup && warmup[0]=='0'))return;
            const auto& sampling=input.sampling;
            const bool wide=sampling.temperature<=0 || sampling.top_k==1 ||
                (sampling.top_k>0 && sampling.top_k<=64);
            const int requested=sampling.enable_mtp && wide
                ?std::clamp(int(sampling.mtp_max_draft_tokens),1,components.mtp->maximum_draft_depth()):1;
            // Short prefill shares the existing four-row graphs. Only draft
            // caps above three need additional verification storage.
            const int depth=std::max(3,requested);
            if(depth==graph_draft_depth && language.graph_registry)return;
            const auto started=std::chrono::steady_clock::now();
            preparing_graph_depth=true;
            try {
                weights.release_workspace();
                graph.invalidate();
                qwen4_release_decode_graphs(language,components.mtp.get());
                release_moe_decode_buffers(::tyloi::weights::detail::cache(weights).moe);
                state.reset(1);components.mtp->reset(1);
                auto context=tyloi::cuda::default_context(execution->config.layer_placement.primary_device());
                context->reclaim_idle_memory();
                budget->reclaim((std::size_t(2)<<30)+qwen4_mtp_graph_reserve(language,components.mtp.get(),depth));
                qwen4_warmup_decode_graphs(language,components.mtp.get(),depth);
                graph_draft_depth=depth;
                preparing_graph_depth=false;
                configure_decode_budget();
                budget->prepare(tyloi::engine::Budget::Stage::loaded, source.get());
            } catch(...) {
                graph_draft_depth=0;preparing_graph_depth=false;
                throw;
            }
            std::cerr<<"runtime_mtp_depth_plan requested_depth="<<requested<<" prepared_depth="<<depth
                <<" ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()<<'\n';
        }
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
            tyloi::engine::RuntimeConfig runtime_config,
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
        budget.emplace(*weight_memory, weights, cache,
            tyloi::Device(tyloi::kCUDA, execution->config.layer_placement.primary_device()));
        if (offload) offload->bind_budget(&*budget);
        if (config.continuous_batch.max_sequences) {
            if constexpr (std::is_same_v<Model,qwen4_exp::Model>)
                batching = make_batch(language, *execution, config.continuous_batch,
                    config.generation.prefill_chunk_size);
            else {
                auto batch_config = config.continuous_batch;
                batch_config.cuda_graph &= model_parallel_cuda_graph_enabled(execution->config, execution->executor(), execution->weights);
                batching = make_batch(
                    std::shared_ptr<const Model>(model), &state, execution->executor().fork(model_parallel_primary_device(execution->config)),
                    execution->executor().collectives(), execution->executor().forward_mode(execution->config.layer_placement.primary_device()),
                    ::tyloi::weights::detail::cache(weights).moe, std::move(batch_config),
                    config.generation.prefill_chunk_size,
                    runtime_options::kv_vram_preallocated(),
                    execution->config.dense_cpu_layer_count != 0 ||
                        !execution->config.dsv4_cpu_offload_layers.empty(),
                    execution->config.weights.min_slots);
            }
        }
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if(::tyloi::weights::detail::cache(weights).moe) {
            auto expert_loan=kv?kv->expert_loan():nullptr;
            if(!expert_loan && (components.mtp || moe_expert_cache_ssd_experts(::tyloi::weights::detail::cache(weights).moe))) {
                expert_loan=std::make_shared<KvExpertLoan>();
                set_moe_kv_expert_loan(::tyloi::weights::detail::cache(weights).moe,expert_loan);
            }
            if(expert_loan)
                ::tyloi::weights::detail::init_workspace(weights,
                    execution->config.layer_placement.primary_device(), std::move(expert_loan));
            bind_moe_budget(::tyloi::weights::detail::cache(weights).moe, &*budget);
        }
        if constexpr (std::is_same_v<Model,qwen4_exp::Model>) {
            language.reserve_kv_workspace = [this](std::size_t bytes) {
                budget->reserve_kv(bytes);
            };
            if(kv)kv->bind_budget(&*budget);
        }
#endif
        if(const auto bytes=runtime_options::resident_budget_bytes()) {
            SessionCommand command;
            command.kind=SessionCommand::Kind::memory_budget;
            command.bytes=bytes;
            session(command);
        }
        if constexpr(std::is_same_v<Model,qwen4_exp::Model>) {
            qwen4_warmup_decode_graphs(language,components.mtp.get());
            // Exercise the same draft/verify/commit loop as a real request.
            // Graph-only warmup cannot cover its history replay, sampling and
            // expert exchange boundaries. No synthetic policy/history survives.
            const auto* warmup=tyloi::cuda::env_lookup("TYLOINFER_QWEN4_WARMUP_MTP");
            if(language.graph_registry && components.mtp &&
                    language.max_position_embeddings()>=48 && (!warmup || warmup[0]!='0')) {
                struct CacheLearningScope {
                    std::shared_ptr<MoeExpertCache> cache;bool previous;
                    explicit CacheLearningScope(std::shared_ptr<MoeExpertCache> value):cache(std::move(value)),
                        previous(set_moe_reuse_learning(cache,false)) {}
                    ~CacheLearningScope(){set_moe_reuse_learning(cache,previous);}
                } learning(::tyloi::weights::detail::cache(weights).moe);
                auto& predictor=*components.mtp;
                const auto saved_policy=mtp_state.policy;
                const auto start=std::chrono::steady_clock::now();
                InferenceRequest request;
                request.chat=false;
                request.sampling.temperature=0.7;
                request.sampling.top_p=0.8;
                request.sampling.top_k=20;
                request.sampling.seed=1234;
                request.sampling.max_tokens=static_cast<int32_t>(
                    std::min<int64_t>(64,language.max_position_embeddings()-16));
                request.sampling.mtp_max_draft_tokens=3;
                for(int i=0;i<16;++i)request.prompt.push_back((i*7919+17)%language.metadata.vocab_size);
                InferenceOutput output(request,nullptr,"cuda-mtp-warmup");
                auto scope=language.execution_scope();
                auto generation=run_mtp_generation(language,state,executor.get(),execution->executor().collectives(),
                                                   execution->executor().forward_mode(execution->config.layer_placement.primary_device()),weights,info,source.get(),predictor,mtp_state,request,output,16,
                                                   nullptr,0,{},nullptr,0,nullptr);
                while(generation.next()) {}
                finish_moe_expert_exchanges(::tyloi::weights::detail::cache(weights).moe);
                predictor.reset(1);
                state.reset(1);
                mtp_state.policy=saved_policy;
                mtp_state.stats={};
                mtp_state.cycles=mtp_state.accepted=mtp_state.rejected=0;
                TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_get_current_cuda_stream()));
                std::cerr<<"runtime_qwen4_mtp_warmup tokens="<<output.result.completion_tokens
                    <<" ms="<<std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now()-start).count()<<'\n';
                release_moe_prefill_buffers(::tyloi::weights::detail::cache(weights).moe);
                tyloi::cuda::default_context(execution->config.layer_placement.primary_device())->reclaim_idle_memory();
            }
        }
        preparing_graph_depth=false;
        configure_decode_budget();
        budget->prepare(tyloi::engine::Budget::Stage::loaded, source.get());
        } catch (...) {
            const auto failure = std::current_exception();
            try {
                drain_cuda(root_executor, weights, offload.get(), execution->config.layer_placement,
                           execution->config.tensor_parallel, execution->config.expert_parallel);
            }
            catch (const std::exception& error) { std::cerr << "runtime_load_cleanup error=" << error.what() << '\n'; }
            clear_bindings();
            std::rethrow_exception(failure);
        }
    }
    bool can_batch(const EngineRequest& request) const {
        if constexpr (std::is_same_v<Model,qwen4_exp::Model>)
            return batching && !request.input.media && request.input.cache_plan.session_id.empty() &&
                !(request.input.sampling.enable_mtp && components.mtp);
        return batching && batch_compatible(request, cache.persistent_prefix_enabled(), bool(components.mtp));
    }
    void clear_bindings() noexcept {
        if constexpr(std::is_same_v<Model,qwen4_exp::Model>) language.reserve_kv_workspace={};
        if(kv)kv->bind_budget(nullptr);
        bind_moe_budget(::tyloi::weights::detail::cache(weights).moe, nullptr);
        if (offload) offload->bind_budget(nullptr);
    }
    ~Text() {
        clear_bindings();
        batching.reset();
    }
    SessionResult session(const SessionCommand& command) {
        if(command.kind!=SessionCommand::Kind::memory_budget)return cache.control(command);
        if(!online_budget_supported())throw std::invalid_argument("online memory budgeting requires a single-device CUDA worker");
        auto change = budget->change(command);
        weights.release_workspace();
        finish_moe_expert_exchanges(::tyloi::weights::detail::cache(weights).moe);
        graph.invalidate();
        if constexpr(std::is_same_v<Model,qwen4_exp::Model>)
            qwen4_release_decode_graphs(language,components.mtp.get());
        if(batching)batching->release_idle_buffers();
        if(components.mtp)components.mtp->reset(1);
        state.reset(1);
        return change.apply();
    }
    bool exclusive() const { return duplex_active; }
    bool mtp_available() const { return bool(components.mtp); }
    void execute(const std::vector<RequestId>& eligible) {
        if (batching) batching->step(eligible);
    }
    using Prepared = PreparedPrompt;
    tyloi::StepSequence<Prepared> prepare(InferenceRequest& input, State& state) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if constexpr (requires { static_cast<const tyloi::models::CausalModel&>(language).prepare(
                                   input.prompt, *input.vision, state); }) {
            const tyloi::models::CausalModel& causal = language;
            return causal.prepare(input.prompt, *input.vision, state);
        } else
#endif
        return components.prepare(language, images,
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
            execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)),
            execution->executor().collectives(),
#endif
            execution->config.layer_placement.primary_device(),
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
            executor.get(),
#endif
            input.prompt, *input.vision);
    }
    auto advance_preparation(tyloi::StepSequence<Prepared>& preparation) {
        PrefillCudaTimer timer;
        auto step = preparation.next();
        TYLOINFER_CUDA_CHECK(cudaEventRecord(timer.finished_event(), mfq_get_current_cuda_stream()));
        return std::pair{std::move(step), timer.elapsed_ms()};
    }
    Generation generate_text(InferenceRequest& input, InferenceOutput& output, State& state,
                             std::optional<Prepared> prepared, const RequestId& id, bool batched) {
        if(!batched)prepare_graph_depth(input);
        engine::Budget* decode_budget = nullptr;
        if (!preparing_graph_depth) configure_decode_budget();
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if (!preparing_graph_depth) decode_budget = &*budget;
#endif
        return internal::generate(
            language, state, executor.get(), execution->executor().collectives(),
            execution->executor().forward_mode(execution->config.layer_placement.primary_device()), execution->executor().profiler(), weights,
            execution->executor().mode().ffn.serial, execution->config.report_cuda_memory,
            execution->config.dsv4_cpu_offload_layers.empty() && execution->config.dense_cpu_layer_count == 0 &&
                model_parallel_cuda_graph_enabled(execution->config, execution->executor(), execution->weights),
            [this] {
                drain_cuda(root_executor, weights, offload.get(), execution->config.layer_placement,
                           execution->config.tensor_parallel, execution->config.expert_parallel);
            },
            info, source.get(), graph, cache, config, input, output,
            mtp_state,
            batched ? nullptr : components.mtp.get(), std::move(prepared),
            batched ? batching.get() : nullptr, id, decode_budget);
    }
    Generation generate(const RequestId& id, InferenceRequest& input, InferenceOutput& output, bool batched) {
        if (batched) {
            graph.invalidate();
            if constexpr (std::is_same_v<Model,qwen4_exp::Model>) {
                // Exclusive MTP generation has ended before a batch starts.
                // Its graph consumers must retire before idle batch buffers.
                qwen4_release_decode_graphs(language,components.mtp.get());
                if (components.mtp) components.mtp->reset(1);
            }
        }
        return generate_prepared(*this, state, input, output, id, batched);
    }
    template <class T> ControlResult control(T value) {
        if constexpr (std::is_same_v<T, RuntimeMetrics>) {
            const auto& model = std::as_const(language);
            const auto& parts = std::as_const(components);
            const auto component_metrics = [&] {
                if constexpr (requires { static_cast<const tyloi::models::CausalModel&>(model); })
                    return static_cast<const tyloi::models::CausalModel&>(model).metrics();
                else
                    return parts.metrics();
            }();
            Metrics profile_metrics;
            if constexpr (requires { parts.profile_metrics(model); })
                profile_metrics = parts.profile_metrics(model);
            std::optional<Metrics> memory;
            if constexpr (requires { parts.memory_metrics(model); })
                memory = parts.memory_metrics(model);
            const bool streamed_kv = parts.streamed_kv();
            const auto* mtp = parts.mtp.get();
            const auto batch_metrics = batching ? batching->metrics() : Metrics{};
            const auto kv_metrics = kv ? kv->metrics() : Metrics{};
            const auto offload_metrics = offload ? std::optional<Metrics>{offload->metrics()} : std::nullopt;
            Metrics predictor_metrics;
            if (mtp) {
                tyloi::engine::mtp::append_generation_metrics(predictor_metrics, mtp_state.stats);
                predictor_metrics.emplace_back("mtp_predictor_graph_replays", double(mtp->graph_replay_count()));
            }
            return tyloi::engine::runtime_metrics(component_metrics,profile_metrics,std::move(memory),streamed_kv,
                graph_draft_depth,batch_metrics,kv_metrics,offload_metrics,predictor_metrics,source.get(),
                *weight_memory,weights,cache,*budget,
                execution->executor().memory(cuda_devices(execution->config)),
                tyloi::Device(tyloi::kCUDA, execution->config.layer_placement.primary_device()),
                execution->config.layer_placement.devices.size() <= 1 &&
                    !execution->config.tensor_parallel.enabled() && !execution->config.expert_parallel.enabled());
        } else {
            auto result = components.control(value);
            if constexpr (std::is_same_v<T, MfqDuplexSessionParams>)
                duplex_active = true;
            else if constexpr (!std::is_same_v<T, MfqDuplexStepInput>)
                duplex_active = false;
            return result;
        }
    }
};

} // namespace tyloi::cuda::engine_detail
