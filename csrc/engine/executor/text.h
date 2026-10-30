#pragma once

#include "env.h"
#include "cuda_runtime_config.h"
#include "backends/cuda/engine/models/components.h"
#include "backends/cuda/engine/weights/routed_cache.h"
#include "backends/cuda/engine/executor/execution.h"
#include "engine/executor/budget.h"
#include "engine/executor/tq.h"
#include "backends/cuda/engine/executor/generation.h"
#include "backends/cuda/engine/weights/load.h"
#include "backends/cuda/engine/weights/moe/expert_cache.h"
#include "backends/cuda/engine/state/kv_offload.h"
#include "backends/cuda/engine/state/kv_memory.h"
#include "backends/cuda/engine/weights/moe/kv_expert_loan.h"
#include "engine/weights/workspace.h"
#include "backends/cuda/engine/executor/options.h"
#include "engine/batch/batch.h"
#include "backends/cuda/engine/state/session/text_cache.h"
#include "engine/executor/metrics.h"
#include "engine/requests/table.h"
#include "engine/executor/flow.h"
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

namespace mfq::cuda::engine_detail {

using namespace mfq::engine;
using namespace mfq::cuda::internal;

template <class M, class S>
std::unique_ptr<engine::Batch> make_batch(
    std::shared_ptr<const M>, S*, std::unique_ptr<Executor>, Collectives&,
    Executor::Mode, std::shared_ptr<MoeExpertCache>, CudaContinuousBatchConfig,
    std::int64_t, bool, bool, int) {
    throw std::invalid_argument("continuous batching is unavailable for this model");
}

template <class LoadedModel>
struct Text {
    using Model = typename LoadedModel::Model;
    using State = typename LoadedModel::State;
    using Parts = typename LoadedModel::Parts;

    std::shared_ptr<WeightMemoryLedger> weight_memory;
    CudaExecutionContext* execution;
    RoutedCache& weights;
    std::shared_ptr<KvMemoryManager>& kv;
    std::shared_ptr<KvOffloadStore>& offload;
    CudaRuntimeConfig config;
    std::shared_ptr<const mfq::ModelSource> source;
    LoadedModel loaded;
    Parts& components;
    std::unique_ptr<Executor>& executor;
    Model& language;
    State& state;
    const mfq::models::CausalLmMetadata& metadata;
    std::function<void()> restore_decode_workspace;
    DecodeGraphCache graph;
    TextSessionCache cache;
    std::optional<Budget> budget;
    std::unique_ptr<mfq::engine::Batch> batching;
    bool duplex_active = false;
    std::optional<ExpertSpace> expert_space;
    int graph_draft_depth=3;
    bool preparing_graph_depth=true;

    void restore_mtp_expert_space() {
        // These regions hold legacy paged expert fields. TQ retains ordinary
        // compressed tensor images and cannot consume those page addresses.
        if(weights.tq)return;
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            if(preparing_graph_depth || !expert_space || expert_space->has_decode() || !components.mtp)return;
            const auto maximum=qwen4_mtp_graph_reserve(language,components.mtp.get());
            const auto active=qwen4_mtp_graph_reserve(language,components.mtp.get(),graph_draft_depth);
            if(active>=maximum)return;
            const auto bytes=expert_space->lend_decode(maximum-active);
            if(bytes)
                std::cerr<<"runtime_mtp_workspace_cache action=lend depth="
                    <<graph_draft_depth<<" bytes="<<bytes<<'\n';
        }
    }

    void restore_graph_expert_space() {
        if(weights.tq)return;
        if(preparing_graph_depth)return;
        prepare_moe_decode_residency(weights.moe);
        mfq::cuda::default_context(execution->layer_placement.primary_device())->reclaim_idle_memory();
        restore_mtp_expert_space();
        if(expert_space)
            expert_space->restore(runtime_options::workspace_reserve_bytes(), source.get());
    }

    void prepare_graph_depth(const InferenceRequest& input) {
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            if(!components.mtp)return;
            const auto* warmup=mfq::cuda::env_lookup("TYLOINFER_QWEN4_WARMUP_GRAPHS");
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
                if(expert_space) expert_space->release();
                graph.invalidate();
                qwen4_release_decode_graphs(language,components.mtp.get());
                release_moe_decode_buffers(weights.moe);
                state.reset(1);components.mtp->reset(1);
                auto context=mfq::cuda::default_context(execution->layer_placement.primary_device());
                context->reclaim_idle_memory();
                budget->reclaim((std::size_t(2)<<30)+qwen4_mtp_graph_reserve(language,components.mtp.get(),depth),
                    expert_space ? &*expert_space : nullptr);
                qwen4_warmup_decode_graphs(language,components.mtp.get(),depth);
                graph_draft_depth=depth;
                preparing_graph_depth=false;
                restore_graph_expert_space();
            } catch(...) {
                graph_draft_depth=0;preparing_graph_depth=false;
                throw;
            }
            std::cerr<<"runtime_mtp_depth_plan requested_depth="<<requested<<" prepared_depth="<<depth
                <<" ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()<<'\n';
        }
    }

    bool online_budget_supported() const {
        return execution->layer_placement.devices.size()<=1 && !execution->tensor_parallel.enabled()
            && !execution->expert_parallel.enabled();
    }

    Text(CudaExecutionContext& borrowed, RoutedCache& borrowed_weights,
            std::shared_ptr<KvMemoryManager>& borrowed_kv,
            std::shared_ptr<KvOffloadStore>& borrowed_offload,
            LoadedModel loaded_model,
            CudaRuntimeConfig runtime_config,
            std::shared_ptr<const mfq::ModelSource> model_source,
            std::shared_ptr<WeightMemoryLedger> ledger)
        : weight_memory(std::move(ledger)), execution(&borrowed), weights(borrowed_weights),
          kv(borrowed_kv), offload(borrowed_offload), config(std::move(runtime_config)),
          source(std::move(model_source)),
          loaded(std::move(loaded_model)), components(loaded.components), executor(loaded.executor),
          language(loaded.language()), state(loaded.state()),
          metadata(language.metadata),
          graph(language.metadata.max_position_embeddings),
          cache(config.session_cache, config.prefix_cache,
              make_cuda_paged_prefix_cache(*source, language.metadata.max_position_embeddings,
                  state.supports_paged_text_session_state(), config.prefix_cache),
                state.supports_text_session_state(), state.supports_text_session_state() ? 0 : 1,
                cuda_has_separate_memory()) {
        try {
        budget.emplace(*weight_memory, weights, cache,
            execution->layer_placement.primary_device());
        if (offload)
            offload->set_ram_reservation([this](std::size_t bytes) {
                cache.reserve_ram(bytes);weights.reserve(bytes);
            });
        if (config.continuous_batch.max_sequences) {
            if constexpr (std::is_same_v<Model,Qwen4CausalLm>)
                batching = make_batch(language, *execution, config.continuous_batch,
                    config.generation.prefill_chunk_size);
            else {
                auto batch_config = config.continuous_batch;
                batch_config.cuda_graph &= model_parallel_cuda_graph_enabled(*execution);
                batching = make_batch(
                    std::shared_ptr<const Model>(loaded.model), &state, execution->make_executor(),
                    execution->collectives, execution->forward_mode(),
                    weights.moe, std::move(batch_config),
                    config.generation.prefill_chunk_size,
                    runtime_options::kv_vram_preallocated(),
                    execution->dense_cpu_layer_count != 0 ||
                        !execution->dsv4_cpu_offload_layers.empty(),
                    execution->config.weights.min_slots);
            }
        }
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if(weights.moe) {
            auto expert_loan=kv?kv->expert_loan():nullptr;
            if(!expert_loan && (components.mtp || moe_expert_cache_ssd_experts(weights.moe))) {
                expert_loan=std::make_shared<KvExpertLoan>();
                set_moe_kv_expert_loan(weights.moe,expert_loan);
            }
            if(expert_loan)
                expert_space.emplace(weights,
                    execution->layer_placement.primary_device(), std::move(expert_loan));
            restore_decode_workspace=[this]{restore_graph_expert_space();};
            if constexpr (requires(Model& value) {
                              value.restore_decode_workspace = std::function<void()>();
                          })
                language.restore_decode_workspace = restore_decode_workspace;
            set_moe_workspace_reservation(weights.moe,
                [this](std::size_t bytes) {
                    budget->reclaim(bytes, expert_space ? &*expert_space : nullptr);
                },
                [this](std::size_t bytes) {
                    cache.reserve_ram(bytes);weights.reserve(bytes);
                });
        }
        if constexpr (std::is_same_v<Model,Qwen4CausalLm>) {
            language.reserve_kv_workspace = [this](std::size_t bytes) {
                const auto device = execution->layer_placement.primary_device();
                auto context = mfq::cuda::default_context(device);
                if (context->available_memory() >= bytes) return;
                if(expert_space && !expert_space->empty()) {
                    std::cerr<<"runtime_workspace_cache_reclaim reason=kv_growth requested_bytes="<<bytes
                        <<" available_bytes="<<context->available_memory()<<'\n';
                    expert_space->release(bytes);
                }
                if(context->available_memory()>=bytes)return;
                std::size_t hot = 0;
                for (const auto& [name,value] : cache.metrics())
                    if (name == "prefix_cache_hot_bytes") hot = std::size_t(value);
                const auto shortage = bytes - context->available_memory();
                cache.trim_hot(hot > shortage ? hot - shortage : 0);
                context->reclaim_idle_memory();
                if (context->available_memory() < bytes)
                    throw InferenceInputError(InferenceInputErrorCode::ContextLength,
                        "insufficient VRAM for context growth; reduce concurrent requests or prompt length");
            };
            if(kv)kv->set_reservation(language.reserve_kv_workspace);
        }
#endif
        if(const auto bytes=runtime_options::resident_budget_bytes()) {
            SessionCommand command;
            command.kind=SessionCommand::Kind::memory_budget;
            command.bytes=bytes;
            session(command);
        }
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            qwen4_warmup_decode_graphs(language,components.mtp.get());
            // Exercise the same draft/verify/commit loop as a real request.
            // Graph-only warmup cannot cover its history replay, sampling and
            // expert exchange boundaries. No synthetic policy/history survives.
            const auto* warmup=mfq::cuda::env_lookup("TYLOINFER_QWEN4_WARMUP_MTP");
            if(language.graph_registry && components.mtp &&
                    language.max_position_embeddings()>=48 && (!warmup || warmup[0]!='0')) {
                struct CacheLearningScope {
                    std::shared_ptr<MoeExpertCache> cache;bool previous;
                    explicit CacheLearningScope(std::shared_ptr<MoeExpertCache> value):cache(std::move(value)),
                        previous(set_moe_reuse_learning(cache,false)) {}
                    ~CacheLearningScope(){set_moe_reuse_learning(cache,previous);}
                } learning(weights.moe);
                auto& predictor=*components.mtp;
                const auto saved_policy=predictor.policy_state;
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
                for(int i=0;i<16;++i)request.prompt.push_back((i*7919+17)%language.vocab_size());
                InferenceOutput output(request,nullptr,"cuda-mtp-warmup");
                auto scope=language.execution_scope();
                auto generation=run_mtp_generation(language,state,executor.get(),execution->collectives,
                                                   execution->forward_mode(),weights,metadata,predictor,request,output,16,
                                                   nullptr,0,{},nullptr,0,restore_decode_workspace);
                while(generation.next()) {}
                finish_moe_expert_exchanges(weights.moe);
                predictor.reset(1);
                state.reset(1);
                predictor.policy_state=saved_policy;
                predictor.last_stats={};
                predictor.last_cycles=predictor.last_accepted=predictor.last_rejected=0;
                TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_get_current_cuda_stream()));
                std::cerr<<"runtime_qwen4_mtp_warmup tokens="<<output.result.completion_tokens
                    <<" ms="<<std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now()-start).count()<<'\n';
                release_moe_prefill_buffers(weights.moe);
                mfq::cuda::default_context(execution->layer_placement.primary_device())->reclaim_idle_memory();
            }
        }
        preparing_graph_depth=false;
        restore_graph_expert_space();
        } catch (...) {
            const auto failure = std::current_exception();
            try { execution->drain(); }
            catch (const std::exception& error) { std::cerr << "runtime_load_cleanup error=" << error.what() << '\n'; }
            clear_bindings();
            std::rethrow_exception(failure);
        }
    }
    bool can_batch(const EngineRequest& request) const {
        if constexpr (std::is_same_v<Model,Qwen4CausalLm>)
            return batching && !request.input.media && request.input.cache_plan.session_id.empty() &&
                !(request.input.sampling.enable_mtp && components.mtp);
        return batching && batch_compatible(request, cache.persistent_prefix_enabled(), bool(components.mtp));
    }
    void clear_bindings() noexcept {
        restore_decode_workspace={};
        if constexpr (requires(Model& value) {
                          value.restore_decode_workspace = std::function<void()>();
                      })
            language.restore_decode_workspace={};
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) language.reserve_kv_workspace={};
        if(kv)kv->set_reservation({});
        set_moe_workspace_reservation(weights.moe,{});
        if (offload) offload->set_ram_reservation({});
    }
    ~Text() {
        expert_space.reset();
        clear_bindings();
    }
    SessionResult session(const SessionCommand& command) {
        if(command.kind!=SessionCommand::Kind::memory_budget)return control_session(cache,command);
        if(!online_budget_supported())throw std::invalid_argument("online memory budgeting requires a single-device CUDA worker");
        return budget->set(command, [this] {
            if(expert_space) expert_space->release();
            finish_moe_expert_exchanges(weights.moe);
            graph.invalidate();
            if constexpr(std::is_same_v<Model,Qwen4CausalLm>)
                qwen4_release_decode_graphs(language,components.mtp.get());
            if(batching)batching->release_idle_buffers();
            if(components.mtp)components.mtp->reset(1);
            state.reset(1);
        });
    }
    bool exclusive() const { return duplex_active; }
    bool mtp_available() const { return bool(components.mtp); }
    void execute(const std::vector<RequestId>& eligible) {
        if (batching) batching->step(eligible);
    }
    using Prepared = PreparedPrompt;
    mfq::StepSequence<Prepared> prepare(InferenceRequest& input) {
        return components.prepare(language, execution->projector(), weights.moe,
            execution->collectives, execution->layer_placement.primary_device(), executor.get(),
            input.prompt, *input.vision);
    }
    auto advance_preparation(mfq::StepSequence<Prepared>& preparation) {
        PrefillCudaTimer timer;
        auto step = preparation.next();
        TYLOINFER_CUDA_CHECK(cudaEventRecord(timer.finished_event(), mfq_get_current_cuda_stream()));
        return std::pair{std::move(step), timer.elapsed_ms()};
    }
    Generation generate_text(InferenceRequest& input, InferenceOutput& output, std::optional<Prepared> prepared,
                             const RequestId& id, bool batched) {
        if(!batched)prepare_graph_depth(input);
        return internal::generate(
            language, state, executor.get(), execution->collectives,
            execution->forward_mode(), execution->profiler, weights,
            execution->decode_graph_serial_branches, execution->config.report_cuda_memory,
            execution->dsv4_cpu_offload_layers.empty() && execution->dense_cpu_layer_count == 0 &&
                model_parallel_cuda_graph_enabled(*execution),
            [owner = execution] { owner->drain(); },
            metadata, graph, cache, config, input, output,
            batched ? nullptr : components.mtp.get(), std::move(prepared),
            batched ? batching.get() : nullptr, id, restore_decode_workspace);
    }
    Generation generate(const RequestId& id, ExecutionRequest& request) {
        if (request.batched) {
            graph.invalidate();
            if constexpr (std::is_same_v<Model,Qwen4CausalLm>) {
                // Exclusive MTP generation has ended before a batch starts.
                // Its graph consumers must retire before idle batch buffers.
                qwen4_release_decode_graphs(language,components.mtp.get());
                if (components.mtp) components.mtp->reset(1);
            }
        }
        return generate_prepared(*this, request.input, request.output, id, request.batched);
    }
    template <class T> ControlResult control(T value) {
        if constexpr (std::is_same_v<T, RuntimeMetrics>) {
            return runtime_metrics(language,components,graph_draft_depth,
                expert_space ? &*expert_space : nullptr,batching.get(),source.get(),
                *weight_memory,weights,kv.get(),offload.get(),cache,*budget,
                execution->layer_placement,execution->tensor_parallel,execution->expert_parallel);
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

} // namespace mfq::cuda::engine_detail
