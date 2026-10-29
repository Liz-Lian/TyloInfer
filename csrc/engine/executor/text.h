#pragma once

#include "env.h"
#include "cuda_runtime_config.h"
#include "backends/cuda/engine/models/components.h"
#include "backends/cuda/engine/weights/routed_cache.h"
#include "backends/cuda/engine/executor/execution.h"
#include "engine/executor/budget.h"
#include "engine/executor/tq.h"
#include "backends/cuda/engine/executor/generation.h"
#include "models/minicpmo45/forward.h"
#include "backends/cuda/engine/weights/load.h"
#include "backends/cuda/engine/weights/moe/expert_cache.h"
#include "backends/cuda/engine/state/kv_offload.h"
#include "backends/cuda/engine/weights/moe/kv_expert_loan.h"
#include "engine/weights/workspace.h"
#include "mfq_cuda_kv_quantization.h"
#include "backends/cuda/engine/executor/options.h"
#include "engine/batch/batch.h"
#include "backends/cuda/engine/state/session/text_cache.h"
#include "engine/executor/mtp_metrics.h"
#include "engine/requests/table.h"
#include "engine/executor/flow.h"
#include "weight_memory.h"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
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

template <class M>
std::optional<KvQuantization> kv_quantization(const M&) {
    return std::nullopt;
}

template <class LoadedModel>
struct Text {
    using Model = typename LoadedModel::Model;
    using State = typename LoadedModel::State;
    using Parts = typename LoadedModel::Parts;

    std::shared_ptr<WeightMemoryLedger> weight_memory;
    CudaExecutionContext* execution;
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
        if(execution->stores.weights.tq)return;
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
        if(execution->stores.weights.tq)return;
        if(preparing_graph_depth)return;
        prepare_moe_decode_residency(execution->stores.weights.moe);
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
                release_moe_decode_buffers(execution->stores.weights.moe);
                state.reset(1);components.mtp->reset(1);
                auto context=mfq::cuda::default_context(execution->layer_placement.primary_device());
                context->reclaim_idle_memory();
                reclaim_prefix_workspace((std::size_t(2)<<30)+qwen4_mtp_graph_reserve(language,components.mtp.get(),depth));
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

    void reclaim_prefix_workspace(std::size_t bytes) {
        auto context=mfq::cuda::default_context(execution->layer_placement.primary_device());
        const auto available=context->available_memory();
        if(available>=bytes)return;
        if(expert_space && !expert_space->empty()) {
            std::cerr<<"runtime_workspace_cache_reclaim reason=workspace requested_bytes="<<bytes
                <<" available_bytes="<<available<<'\n';
            expert_space->release(bytes);
        }
        if(context->available_memory()>=bytes)return;
        std::size_t hot=0;
        for(const auto& [name,value]:cache.metrics())
            if(name=="prefix_cache_hot_bytes")hot=std::size_t(value);
        const auto shortage=bytes-context->available_memory();
        const auto released=cache.trim_hot(hot>shortage?hot-shortage:0);
        context->reclaim_idle_memory();
        if(released)std::cerr<<"runtime_workspace_reclaim requested_bytes="<<bytes
            <<" available_bytes="<<context->available_memory()<<" prefix_released_bytes="<<released<<'\n';
    }

    Text(CudaExecutionContext& borrowed, LoadedModel loaded_model,
            CudaRuntimeConfig runtime_config,
            std::shared_ptr<const mfq::ModelSource> model_source,
            std::shared_ptr<WeightMemoryLedger> weights)
        : weight_memory(std::move(weights)), execution(&borrowed), config(std::move(runtime_config)),
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
        budget.emplace(*weight_memory, execution->stores.weights, cache,
            execution->layer_placement.primary_device());
        if (execution->stores.qsa_kv_store)
            execution->stores.qsa_kv_store->set_ram_reservation([this](std::size_t bytes) {
                cache.reserve_ram(bytes);execution->stores.weights.reserve(bytes);
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
                    execution->stores.weights.moe, std::move(batch_config),
                    config.generation.prefill_chunk_size,
                    runtime_options::kv_vram_preallocated(),
                    execution->dense_cpu_layer_count != 0 ||
                        !execution->dsv4_cpu_offload_layers.empty(),
                    execution->config.weights.min_slots);
            }
        }
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if(execution->stores.weights.moe) {
            auto expert_loan=execution->stores.kv_memory?execution->stores.kv_memory->expert_loan():nullptr;
            if(!expert_loan && (components.mtp || moe_expert_cache_ssd_experts(execution->stores.weights.moe))) {
                expert_loan=std::make_shared<KvExpertLoan>();
                set_moe_kv_expert_loan(execution->stores.weights.moe,expert_loan);
            }
            if(expert_loan)
                expert_space.emplace(execution->stores.weights,
                    execution->layer_placement.primary_device(), std::move(expert_loan));
            restore_decode_workspace=[this]{restore_graph_expert_space();};
            if constexpr (requires(Model& value) {
                              value.restore_decode_workspace = std::function<void()>();
                          })
                language.restore_decode_workspace = restore_decode_workspace;
            set_moe_workspace_reservation(execution->stores.weights.moe,
                [this](std::size_t bytes) {reclaim_prefix_workspace(bytes);},
                [this](std::size_t bytes) {
                    cache.reserve_ram(bytes);execution->stores.weights.reserve(bytes);
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
            if(execution->stores.kv_memory)execution->stores.kv_memory->set_reservation(language.reserve_kv_workspace);
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
                } learning(execution->stores.weights.moe);
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
                                                   execution->forward_mode(),execution->stores.weights,metadata,predictor,request,output,16,
                                                   nullptr,0,{},nullptr,0,restore_decode_workspace);
                while(generation.next()) {}
                finish_moe_expert_exchanges(execution->stores.weights.moe);
                predictor.reset(1);
                state.reset(1);
                predictor.policy_state=saved_policy;
                predictor.last_stats={};
                predictor.last_cycles=predictor.last_accepted=predictor.last_rejected=0;
                TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_get_current_cuda_stream()));
                std::cerr<<"runtime_qwen4_mtp_warmup tokens="<<output.result.completion_tokens
                    <<" ms="<<std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now()-start).count()<<'\n';
                release_moe_prefill_buffers(execution->stores.weights.moe);
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
        if(execution->stores.kv_memory)execution->stores.kv_memory->set_reservation({});
        set_moe_workspace_reservation(execution->stores.weights.moe,{});
        if (execution->stores.qsa_kv_store) execution->stores.qsa_kv_store->set_ram_reservation({});
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
            finish_moe_expert_exchanges(execution->stores.weights.moe);
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
        if constexpr (requires { components.composite; })
            if (components.composite) return components.composite->prepare(input.prompt, *input.vision);
        if (components.grid_vision)
            return components.grid_vision->prepare(language, execution->projector(),
                execution->stores.weights.moe, execution->collectives,
                execution->layer_placement.primary_device(), executor.get(), input.prompt, *input.vision);
        if constexpr(std::is_same_v<Model,DeepseekV41CausalLm>)
            if(components.deepseek_vision)return components.deepseek_vision->prepare(language,execution->projector(),input.prompt,*input.vision);
        if constexpr(std::is_same_v<Model,NemotronHCausalLm>)
            if (components.radio_vision) return components.radio_vision->prepare(language, execution->projector(), input.prompt, *input.vision);
        throw std::invalid_argument("model has no multimodal component");
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
            execution->forward_mode(), execution->profiler, execution->stores.weights,
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
            const auto quantization = [&]() -> std::optional<KvQuantization> {
                if constexpr (std::is_same_v<Model,Qwen4CausalLm>)
                    return cuda_kv_quantization();
                else
                    return kv_quantization(language);
            }();
            const auto batch_metrics = batching ? batching->metrics() : Metrics{};
            const auto capabilities = components.state();
            const auto workspace_metrics = expert_space ? expert_space->metrics() : ExpertSpace::Metrics{};
            Metrics metrics{
                {"mtp_graph_prepared_depth",double(graph_draft_depth)},
                {"mtp_flexible_expert_capacity_bytes",double(workspace_metrics.decode_capacity)},
                {"mtp_flexible_expert_used_bytes",double(workspace_metrics.decode_used)},
                {"prefill_workspace_expert_capacity_bytes",double(workspace_metrics.prefill_capacity)},
                {"prefill_workspace_expert_used_bytes",double(workspace_metrics.prefill_used)},
                {"prefill_workspace_expert_reclaimed_used_bytes",double(workspace_metrics.prefill_reclaimed_used)},
                {"kv_memory_supported",std::is_same_v<Model,Qwen4CausalLm>?1.0:0.0},
                {"kv_quantization_supported",double(quantization.has_value())},
                {"kv_quantization_bits",quantization?quantization->bits:0.0},
                {"vision_declared", double(capabilities.vision_declared)}, {"vision_supported", double(capabilities.vision_supported)},
                {"vision_available", double(capabilities.vision_available)}, {"mtp_declared", double(capabilities.mtp_declared)},
                {"mtp_supported", double(capabilities.mtp_supported)}, {"mtp_available", double(capabilities.mtp_available)}};
            auto experts = execution->stores.weights.metrics();
            metrics.insert(metrics.end(), experts.begin(), experts.end());
            if constexpr (std::is_same_v<Model,Qwen4CausalLm>) {
                auto phases=qwen4_decode_graph_profile_metrics(language,components.mtp.get());
                metrics.insert(metrics.end(),phases.begin(),phases.end());
            }
            if(source) {
                const auto io=source->file_read_stats();
                metrics.insert(metrics.end(),{{"model_io_direct",io.mode==mfq::FileReadMode::Direct?1.0:0.0},
                    {"model_io_read_calls",double(io.calls)},{"model_io_logical_bytes",double(io.logical_bytes)},
                    {"model_io_physical_bytes",double(io.physical_bytes)},{"model_io_read_ms",double(io.read_nanoseconds)/1e6},
                    {"model_io_staging_bytes",double(io.staging_bytes)},{"model_io_cached_staging_bytes",double(io.staging_cached_bytes)},
                    {"model_io_staging_allocations",double(io.staging_allocations)}});
            }
            if(execution->stores.kv_memory) {
                auto kv=execution->stores.kv_memory->metrics();
                metrics.insert(metrics.end(),kv.begin(),kv.end());
            }
            std::size_t resident_weights=weight_memory->resident_bytes();
            for(const auto& metric:experts)
                if(metric.first=="resident_expert_cache_bytes")resident_weights+=std::size_t(metric.second);
            const auto ple=weight_memory->ple_bytes();
            metrics.insert(metrics.end(),{{"resident_weight_bytes",double(resident_weights)},
                {"ssd_ple_enabled",ple>0?1.0:0.0},{"ssd_ple_payload_bytes",double(ple)}});
            if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
                auto cache_memory=qwen4_cache_memory_metrics(language,components.mtp.get());
                for (auto& [name,value] : cache_memory)
                    for (const auto& [batch_name,batch_value] : batch_metrics) {
                        if (name == "kv_cache_bytes" && batch_name == "continuous_batching_request_kv_bytes")
                            value += batch_value;
                        if (name == "kv_cache_contexts" && batch_name == "continuous_batching_request_contexts")
                            value += batch_value;
                    }
                metrics.insert(metrics.end(),cache_memory.begin(),cache_memory.end());
            }
            const auto metric_bytes = [&](const std::string& key) {
                for(const auto& [name,value]:metrics)if(name==key)return std::size_t(value);
                return std::size_t(0);
            };
            std::set<int> devices(execution->layer_placement.devices.begin(),execution->layer_placement.devices.end());
            devices.insert(execution->tensor_parallel.devices.begin(),execution->tensor_parallel.devices.end());
            devices.insert(execution->expert_parallel.devices.begin(),execution->expert_parallel.devices.end());
            const auto primary=execution->layer_placement.primary_device();
            devices.insert(primary);
            auto prefix_bytes=cache.device_hot_bytes();
            std::size_t allocated=0,reserved=0,free_bytes=0,total_bytes=0,buffers=0;
            for(const int device:devices) {
                MfqCudaGuard guard(device);
                size_t free=0,total=0;
                TYLOINFER_CUDA_CHECK(cudaMemGetInfo(&free,&total));
                const auto memory=mfq_cuda_memory_stats(device);
                const auto external=device==primary?metric_bytes("kv_memory_vmm_bytes"):0;
                const auto resident=std::max(memory.reserved_bytes,memory.allocated_bytes)+external;
                const auto stem="gpu_"+std::to_string(device)+"_";
                auto weights=weight_memory->resident_bytes(device);
                if(device==primary)weights+=metric_bytes("resident_expert_cache_bytes");
                const auto hot=prefix_bytes[device];
                auto live=metric_bytes(stem+"live_kv_bytes");
                if(device==primary)for(const auto& [name,value]:batch_metrics)
                    if(name=="continuous_batching_request_kv_bytes")live+=std::size_t(value);
                metrics.insert(metrics.end(),{{stem+"resident_bytes",double(resident)},
                    {stem+"resident_weight_bytes",double(weights)}, {stem+"prefix_cache_bytes",double(hot)},
                    {stem+"memory_total_bytes",double(total)}, {stem+"memory_available_bytes",double(free)}});
                if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
                    const auto categorized=weights+live+hot;
                    const auto buffer=resident>categorized?resident-categorized:0;
                    metrics.emplace_back(stem+"resident_buffer_bytes",double(buffer));
                    metrics.emplace_back(stem+"kv_bytes",double(live+hot));
                    buffers+=buffer;
                }
                allocated+=memory.allocated_bytes+external;reserved+=memory.reserved_bytes+external;free_bytes+=free;total_bytes+=total;
            }
            metrics.insert(metrics.end(),{{"device_free_bytes",double(free_bytes)}, {"device_total_bytes",double(total_bytes)},
                {"cuda_allocated_bytes",double(allocated)}, {"cuda_reserved_bytes",double(reserved)}});
            if constexpr(std::is_same_v<Model,Qwen4CausalLm>)metrics.emplace_back("resident_buffer_bytes",double(buffers));
            const auto expert_bytes=metric_bytes("resident_expert_cache_bytes");
            const auto minimum=moe_expert_cache_minimum_bytes(execution->stores.weights.moe);
            metrics.insert(metrics.end(),{{"resident_memory_dynamic_budget",online_budget_supported()?1.0:0.0},
                {"resident_weight_dynamic_budget",online_budget_supported()?1.0:0.0},
                {"resident_memory_budget_bytes",double(budget->resident())}, {"resident_weight_budget_bytes",double(budget->weights())},
                {"resident_memory_used_bytes",double(std::max(allocated,reserved))},
                {"resident_reclaimable_weight_bytes",double(expert_bytes>minimum?expert_bytes-minimum:0)}});
            metrics.emplace_back("qsa_kv_offload_supported",std::is_same_v<Model, Qwen4CausalLm> ? 1.0 : 0.0);
            if (execution->stores.qsa_kv_store) {
                auto kv = execution->stores.qsa_kv_store->metrics();
                metrics.insert(metrics.end(), kv.begin(), kv.end());
            } else metrics.insert(metrics.end(), {{"qsa_kv_offload_enabled", 0.0},
                {"qsa_kv_ram_enabled", 0.0}, {"qsa_kv_ram_bytes", 0.0}, {"qsa_kv_ssd_bytes", 0.0}});
            if (components.mtp) {
                mtp::append_generation_metrics(metrics, components.mtp->last_stats);
                metrics.emplace_back("mtp_predictor_graph_replays",double(components.mtp->graph_replay_count()));
            }
            if (batching) {
                metrics.insert(metrics.end(), batch_metrics.begin(), batch_metrics.end());
            }
            return metrics;
        } else {
            if constexpr (requires { components.composite; }) {
                if (!components.composite) throw std::invalid_argument("model has no duplex component");
                if constexpr (std::is_same_v<T, MfqDuplexSessionParams>) {
                    components.composite->start(value); duplex_active = true;
                } else if constexpr (std::is_same_v<T, MfqDuplexStepInput>) {
                    return components.composite->step(value);
                } else { components.composite->stop(); duplex_active = false; }
                return std::monostate{};
            } else {
                throw std::invalid_argument("model has no duplex component");
            }
        }
    }
};

} // namespace mfq::cuda::engine_detail
