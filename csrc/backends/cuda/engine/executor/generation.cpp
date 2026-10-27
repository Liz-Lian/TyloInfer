#include "env.h"
#include "backends/cuda/engine/weights/routed_cache.h"
#include "backends/cuda/engine/executor/generation.h"
#include "engine/executor/flow.h"
#include "backends/cuda/engine/executor/token_row_prefetch.h"
#include "models/deepseek_v4/ops.h"
#include "models/deepseek_v41/ops.h"
#include "models/gemma4/ops.h"
#include "models/glm5_next/ops.h"
#include "models/glm_dsa/ops.h"
#include "models/minicpmo45/ops.h"
#include "models/nemotron_h/ops.h"
#include "models/qwen35/forward.h"
#include "models/qwen4_exp/ops.h"
#include "models/common/full_block.h"
#include "models/common/block.h"
#include "engine/executor/sampler.h"
#include "backends/cuda/engine/state/session/text_cache.h"
#include "mfq_cuda_sampling_ops.h"
#include "backends/cuda/engine/weights/moe/expert_cache.h"
#include "backends/cuda/engine/executor/options.h"
#include "engine/batch/prefill_budget.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>

namespace mfq::cuda::internal {
using mfq::engine::Generation;
namespace {

using Tensor = mfq_tensor_backend::Tensor;

template <typename Model, typename State>
constexpr bool has_forward = requires(
    Model& model, State& state, mfq::cuda::Executor& executor,
    mfq::cuda::Collectives& collectives, const mfq::cuda::Executor::Mode& mode,
    mfq::cuda::Input input) {
    forward(model, state, executor, collectives, mode, std::move(input));
};

template <typename Model>
constexpr bool has_logits = requires(
    Model& model, mfq::cuda::Executor& executor, Tensor hidden) {
    logits(model, executor, std::move(hidden), true);
};

template <typename Model, typename State>
Tensor hidden(Model& model, State& state, mfq::cuda::Executor* executor,
              CudaExecutionContext& execution, mfq::cuda::Input input) {
    static_assert(has_forward<Model, State>);
    TYLOINFER_RUNTIME_CHECK(executor != nullptr,
                            "CUDA model forward requires an executor");
    CudaScope scope(execution);
    return forward(model, state, *executor, execution.collectives,
                   execution.forward_mode(), std::move(input));
}

template <typename Model>
Tensor project(Model& model, mfq::cuda::Executor* executor,
               CudaExecutionContext& execution, Tensor value, bool capped) {
    static_assert(has_logits<Model>);
    TYLOINFER_RUNTIME_CHECK(executor != nullptr,
                            "CUDA model logits requires an executor");
    CudaScope scope(execution);
    return logits(model, *executor, std::move(value), capped);
}

template <typename Model, typename State>
Tensor last_logits(Model& model, State& state, mfq::cuda::Executor* executor,
                   CudaExecutionContext& execution, Tensor ids, bool capped) {
    auto value = hidden(model, state, executor, execution,
                        mfq::cuda::Input{std::move(ids)});
    return project(model, executor, execution,
                   value.index({Slice(), -1, Slice()})
                          .to(mfq_tensor_backend::kFloat16)
                          .contiguous(),
                      capped);
}

template <typename Model, typename State>
static Tensor sample_token(Model &model, State& state, mfq::cuda::Executor* executor,
    CudaExecutionContext &execution, Tensor ids,
    mfq::cuda::Sampler &sampler, mfq_tensor_backend::Tensor counts,
    const MfqTokenConstraintPtr &token_constraint) {
    const auto run = [&]() {
        if (sampler.greedy() && !sampler.has_penalties() && !token_constraint) {
            if constexpr (has_forward<Model, State> && has_logits<Model>) {
                auto logits = last_logits(model, state, executor, execution,
                                              std::move(ids), false)
                                  .contiguous().view({1, -1});
                return sampler.ops().sample_greedy(std::move(logits));
            } else {
                return model.next_token(ids);
            }
        }
        if constexpr (has_forward<Model, State> && has_logits<Model>) {
            auto logits = last_logits(model, state, executor, execution,
                                          std::move(ids), true)
                              .contiguous().view({1, -1});
            return mfq::cuda::sample_logits(sampler, std::move(logits), counts,
                                            token_constraint);
        } else {
            auto logits = model.last_logits(ids).contiguous().view({1, -1});
            return mfq::cuda::sample_logits(sampler, std::move(logits), counts, token_constraint);
        }
    };
    const auto* setting = mfq::cuda::env_lookup("TYLOINFER_PROFILE_DECODE");
    if (!setting || setting[0] != '1' || ids.numel() > 8) return run();
    auto& profiler = execution.profiler;
    const bool previous = profiler.enabled;
    profiler.reset();
    profiler.enabled = true;
    try {
        auto next = profiler.measure("runtime.decode", run);
        profiler.report("runtime.decode");
        profiler.enabled = previous;
        return next;
    } catch (...) {
        profiler.enabled = previous;
        throw;
    }
}

template <typename Model, typename State> struct CudaGenerationOps {
    Model &model;
    State &state;
    mfq::cuda::Executor* executor;
    CudaExecutionContext &execution;
    const mfq::models::CausalLmMetadata &metadata;
    DecodeGraphCache &graph;
    const MfqTokenConstraintPtr &constraint;
    mfq_tensor_backend::Tensor pending, counts;
    mfq_tensor_backend::Tensor known_token_host;
    mfq::cuda::Sampler sampler;
    bool has_penalties;
    const CudaPreparedPrompt *prepared;
    const CudaDecodeGraphConfig &graph_config;
    mfq_tensor_backend::Tensor full_ids;
    std::int32_t generation_limit = 0;
    bool graph_active = false;
    bool graph_prepared = false;
    int64_t prefill_compute_chunk=1024;
    mfq::engine::InferenceOutput* output=nullptr;
    std::function<void()> restore;

    CudaGenerationOps(Model &model, State &state, mfq::cuda::Executor* executor,
        CudaExecutionContext &execution,
        const mfq::models::CausalLmMetadata &metadata, DecodeGraphCache &graph,
        const mfq::engine::InferenceRequest &request, const CudaPreparedPrompt *prepared,
        const CudaDecodeGraphConfig &graph_config,int64_t compute_chunk=1024,
        mfq::engine::InferenceOutput* inference_output=nullptr,
        std::function<void()> restore = {})
        : model(model), state(state), executor(executor), execution(execution), metadata(metadata), graph(graph), constraint(request.token_constraint),
          sampler(request.sampling),
          has_penalties(sampler.has_penalties()), prepared(prepared), graph_config(graph_config),
          prefill_compute_chunk(compute_chunk),output(inference_output),restore(std::move(restore)) {
        full_ids = mfq_tensor_backend::tensor(request.prompt,
            mfq_tensor_backend::TensorOptions()
                .dtype(mfq_tensor_backend::kInt64)
                .device(mfq_tensor_backend::kCUDA))
                       .reshape({1, -1})
                       .contiguous();
        graph.ensure_storage(metadata.vocab_size);
        if (has_penalties) {
            counts = mfq::cuda::SamplingOps::token_counts(full_ids, metadata.vocab_size, graph.counts);
        }
        generation_limit = request.sampling.max_tokens;
        graph_active = graph_eligible();
    }

    double prefill(const mfq::engine::PrefillChunk &chunk) {
        using Tensor = mfq_tensor_backend::Tensor;
        PrefillCudaTimer timer;
        bool prefill_finished=false;
        const auto finish_prefill=[&] {
            if(prefill_finished)return;
            TYLOINFER_CUDA_CHECK(cudaEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
            prefill_finished=true;
        };
        auto ids = full_ids.narrow(1, chunk.offset, chunk.count).contiguous();
        CudaImageMaskScope image_scope(model,prepared,chunk.offset,chunk.count);
        Tensor hidden;
        bool layer_major=false,short_windows=false;
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            // TQ groups all prompt rows by expert without expanding an entire
            // layer. Splitting these batches into verification windows would
            // repeatedly transfer the same expert when it is not resident.
            const bool verification_windows=!execution.stores.tq_expert_cache || chunk.count<=8;
            if(verification_windows && (!prepared || !prepared->transformed()) && model.reuse_decode_graphs &&
                    model.graph_registry && chunk.count<=128) {
                // The same bounded prompt windows as MTP avoid materializing
                // a whole expert layer before a short prompt has any scratch.
                short_windows=true;
                for(int64_t offset=0;offset<chunk.count;) {
                    if(output && output->stopped())break;
                    const auto rows=state.cache_pos==0?int64_t(1):std::min<int64_t>(4,chunk.count-offset);
                    auto part=ids.narrow(1,offset,rows);
                    Tensor window_logits;
                    if(auto result=qwen4_decode_graph_hidden(model,part,nullptr,rows>1?1:0,
                            nullptr,{},&window_logits))
                        hidden=std::move(*result);
                    else hidden=model.hidden_forward(part,mfq_nullopt,mfq_nullopt,nullptr,
                        mfq_nullopt,nullptr,rows>1?1:0);
                    if(rows>1)state.commit_speculative();
                    offset+=rows;
                }
            } else if((!prepared || !prepared->transformed()) &&
                qwen4_layer_major_prefill_eligible(model,chunk.count)) {
                layer_major=true;
                hidden=qwen4_layer_major_prefill_hidden(model,ids,prefill_compute_chunk,true,
                    [&] {return output && output->stopped();},nullptr,false,finish_prefill);
            }
        }
        if constexpr(std::is_same_v<Model,DeepseekV41CausalLm>) {
            if(!prepared || !prepared->transformed())
                hidden=deepseek_v41_prefill_hidden(model,ids,prefill_compute_chunk,
                    [&] {return output && output->stopped();});
        }
        if(output && output->stopped()) {
            finish_prefill();
            return timer.elapsed_ms();
        }
        if (hidden.defined()) {
            // The layer-major path has already advanced every physical cache.
        } else if (prepared && prepared->transformed()) {
            MoePrefillWorkspaceLoan workspace(execution.stores.moe_expert_cache);
            bool borrowed=false;
            if constexpr(std::is_same_v<Model,Qwen4CausalLm>)
                borrowed=workspace.begin((std::size_t(4)<<30)+
                    std::size_t(chunk.count)*model.hidden_size()*model.hc_mult()*sizeof(float)*4)>0;
            if constexpr (has_forward<Model, State>) {
                mfq::cuda::Input input;
                input.ids = ids;
                input.embeddings = prepared->embeddings.narrow(1, chunk.offset, chunk.count)
                                        .contiguous();
                if (prepared->positions.defined())
                    input.positions = prepared->positions.narrow(-1, chunk.offset, chunk.count)
                                           .contiguous();
                input.advance = true;
                hidden = ::mfq::cuda::internal::hidden(
                    model, state, executor, execution, std::move(input));
            } else {
                hidden = model.hidden_forward_inputs(ids,
                    prepared->embeddings.narrow(1, chunk.offset, chunk.count).contiguous(),
                    prepared->positions.defined()?MfqOptional<Tensor>(
                        prepared->positions.narrow(-1,chunk.offset,chunk.count).contiguous()):mfq_nullopt,
                    mfq_nullopt,
                    nullptr,
                    mfq_nullopt,
                    true);
            }
            finish_prefill();
            if(borrowed) {
                workspace.suspend();
                hidden=hidden.narrow(1,hidden.size(1)-1,1).clone();
                workspace.finish();
            }
            state.decode_position_delta = prepared->decode_position_delta;
        } else {
            if constexpr (has_forward<Model, State>) {
                hidden = ::mfq::cuda::internal::hidden(
                    model, state, executor, execution, mfq::cuda::Input{std::move(ids)});
            } else {
                MfqOptional<Tensor> length = mfq_nullopt;
                if (model.adapter_uses_decode_sequence_length() && chunk.offset > 0 && chunk.count == 1)
                    length = mfq_tensor_backend::full({1}, chunk.offset + 1, full_ids.options());
                hidden = model.hidden_forward(ids, mfq_nullopt, length);
            }
        }
        // Match prompt-progress timing: first-token head/sampling and expert
        // restoration remain in TTFT, outside reported prefill throughput.
        finish_prefill();
        if (chunk.offset + chunk.count == full_ids.size(1)) {
            auto input = hidden.index({Slice(), -1, Slice()})
                             .to(mfq_tensor_backend::kFloat16)
                             .contiguous();
            if constexpr (has_logits<Model>) {
                auto logits = project(model, executor, execution, std::move(input), true)
                                  .reshape({1, -1});
                pending = mfq::cuda::sample_logits(sampler, std::move(logits), counts, constraint);
            } else {
                auto logits = model.logits_from_hidden(std::move(input)).reshape({1, -1});
                pending = mfq::cuda::sample_logits(sampler, std::move(logits), counts, constraint);
            }
        }
        const auto elapsed=timer.elapsed_ms();
        if(graph_config.trace)std::cerr<<"runtime_prefill_chunk offset="<<chunk.offset
            <<" tokens="<<chunk.count<<" layer_major="<<layer_major
            <<" short_windows="<<short_windows
            <<" compute_chunk="<<prefill_compute_chunk<<" elapsed_ms="<<elapsed
            <<" timing=before_cache_restore"<<'\n';
        return elapsed;
    }
    std::int64_t first_token() {
        const auto token=pending.template item<int64_t>();
        // Prefill temporaries have retired; expert-cache maintenance remains
        // before token publication, inside TTFT for every CUDA model family.
        if (restore) restore();
        return token;
    }

    bool graph_eligible() const {
        return graph_config.enabled && (!prepared || !prepared->transformed()) && !constraint &&
               !metadata.flash_next && mfq_cuda_graph_capture_supported() &&
               execution.dsv4_cpu_offload_layers.empty() && execution.dense_cpu_layer_count == 0 &&
               !execution.stores.moe_expert_cache && model_parallel_cuda_graph_enabled(execution) &&
               generation_limit >= graph_config.minimum_generation_tokens &&
               generation_limit <= graph.generated_capacity;
    }
    void prepare_graph() {
        using Tensor = mfq_tensor_backend::Tensor;
        graph.ensure_compute_streams(execution.tensor_parallel.enabled()
                                         ? execution.tensor_parallel
                                         : execution.expert_parallel);
        MfqCudaGuard graph_device_guard(graph.stream.device_index());
        auto graph_stream_guards = activate_cuda_graph_compute_streams(graph.compute_streams);
        cudaStream_t graph_stream = graph.stream.stream();
        if (has_penalties) {
            sample_token_counts_add_cuda(counts, pending.contiguous());
        }

        std::int64_t position = state.cache_pos;
        std::int64_t length = position + 1;
        std::int64_t step = 1;
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(graph.static_input.template data_ptr<int64_t>(),
            pending.template data_ptr<int64_t>(),
            sizeof(int64_t),
            cudaMemcpyDeviceToDevice,
            graph_stream));
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(graph.generated.template data_ptr<int64_t>(),
            pending.template data_ptr<int64_t>(),
            sizeof(int64_t),
            cudaMemcpyDeviceToDevice,
            graph_stream));
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(graph.static_pos.template data_ptr<int64_t>(),
            &position,
            sizeof(int64_t),
            cudaMemcpyHostToDevice,
            graph_stream));
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(graph.static_len.template data_ptr<int64_t>(),
            &length,
            sizeof(int64_t),
            cudaMemcpyHostToDevice,
            graph_stream));
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(graph.static_step.template data_ptr<int64_t>(),
            &step,
            sizeof(int64_t),
            cudaMemcpyHostToDevice,
            graph_stream));
        *sampler.ops().random_host().template data_ptr<float>() = 0.5f;
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(graph.random.template data_ptr<float>(),
            sampler.ops().random_host().template data_ptr<float>(),
            sizeof(float),
            cudaMemcpyHostToDevice,
            graph_stream));
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(graph_stream));

        const std::int64_t requested = state.cache_pos + generation_limit;
        const std::int64_t planned =
            decode_graph_bucket(requested, metadata.max_position_embeddings);
        const std::int64_t parts =
            decode_graph_attention_parts(planned, FullBlock::kDecodeAttentionMaxParts);
        const bool greedy = sampler.greedy();
        const auto sample_static = [&]() {
            if constexpr (has_forward<Model, State> && has_logits<Model>) {
                mfq::cuda::Input input;
                input.ids = graph.static_input;
                input.positions = graph.static_pos;
                input.cache_positions = graph.static_pos;
                input.lengths = graph.static_len;
                input.kv_length = planned;
                input.attention_parts = parts;
                auto value = ::mfq::cuda::internal::hidden(
                    model, state, executor, execution, std::move(input));
                auto logits = project(
                    model, executor, execution,
                    value.index({Slice(), -1, Slice()})
                        .to(mfq_tensor_backend::kFloat16)
                        .contiguous(),
                    !(greedy && !has_penalties))
                                  .contiguous().view({1, -1});
                if (has_penalties)
                    logits = sampler.apply_penalties(std::move(logits), counts);
                if (greedy)
                    return sampler.ops().sample_greedy(std::move(logits));
                return sampler.ops().sample_stochastic(
                    std::move(logits), graph.random, sampler.params());
            } else {
                if (greedy && !has_penalties) {
                    return model.next_token_static(
                        graph.static_input, graph.static_pos, graph.static_len, {planned, parts});
                }
                auto logits = model.last_logits_static(
                                      graph.static_input, graph.static_pos, graph.static_len,
                                      {planned, parts})
                                  .contiguous()
                                  .view({1, -1});
                if (has_penalties) {
                    logits = sampler.apply_penalties(std::move(logits), counts);
                }
                if (greedy) {
                    return sampler.ops().sample_greedy(std::move(logits));
                }
                return sampler.ops().sample_stochastic(
                    std::move(logits), graph.random, sampler.params());
            }
        };
        const auto current_graph_storage = [&]() {
            std::vector<Tensor> result;
            const auto append = [&result](const Tensor& tensor) {
                if (tensor.defined()) result.push_back(tensor);
            };
            if constexpr (requires(const State& value) { value.storage(); }) {
                for (const auto& tensor : state.storage()) append(tensor);
            }
            if (executor) {
                for (const auto& attention : executor->scratch()) {
                    append(attention.partial_o);
                    append(attention.partial_m);
                    append(attention.partial_l);
                    append(attention.mask);
                    append(attention.kv_max);
                    append(attention.meta);
                }
            }
            return result;
        };
        auto captured_storage = current_graph_storage();
        const bool hit = graph.matches(planned, sampler.params(), greedy, captured_storage);
        if (hit) {
            ++graph.reuses;
        } else {
            graph.invalidate();
            DecodeGraphBranchScope branch_scope(execution.decode_graph_serial_branches);
            graph.graph = std::make_unique<MfqCudaGraph>();
            try {
                const bool double_warmup = metadata.decode_graph_double_warmup;
                std::vector<Tensor*> warmup_states;
                if constexpr (requires(State& value) { value.graph_state(); }) {
                    warmup_states = state.graph_state();
                } else {
                    for (const auto& block : state.blocks) {
                        for (auto* state : block->graph_warmup_state()) {
                            warmup_states.push_back(state);
                        }
                    }
                }
                prepare_decode_graph_memory(warmup_states, state.cache_pos,
                    double_warmup, *graph.graph, sample_static,
                    graph.graph_participant_streams(execution.collectives));
                graph.graph->capture_begin();
                graph.static_next = sample_static();
                if (has_penalties) {
                    sample_token_counts_add_cuda(counts, graph.static_next.contiguous());
                }
                decode_graph_commit_cuda(graph.static_next,
                    graph.generated,
                    graph.static_step,
                    graph.static_input,
                    graph.static_pos,
                    graph.static_len);
                graph.graph->capture_end();
                captured_storage = current_graph_storage();
                graph.set_key(planned, sampler.params(), greedy, captured_storage);
                ++graph.captures;
            } catch (...) {
                graph.invalidate();
                throw;
            }
        }
        if (!hit)
            report_cuda_memory(execution.config, "runtime_graph_capture");
        if (graph_config.trace) {
            std::cerr << "runtime_cuda_graph action=" << (hit ? "reuse" : "capture")
                      << " requested_len=" << requested << " planned_len=" << planned
                      << " captures=" << graph.captures << " reuses=" << graph.reuses << std::endl;
        }
    }
    std::int64_t advance() {
        if (!graph_active) {
            if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
                if(known_token_host.defined()) {
                    const bool greedy=sampler.greedy() && !has_penalties && !constraint;
                    if(auto result=qwen4_decode_graph_logits(model,pending.reshape({1,1}),greedy?2:1,known_token_host)) {
                        pending=greedy?std::move(*result):mfq::cuda::sample_logits(sampler,
                            result->contiguous().view({1,-1}),counts,constraint);
                        return pending.template item<int64_t>();
                    }
                }
            }
            pending = sample_token(model, state, executor, execution,
                                   pending.reshape({1, 1}), sampler, counts, constraint);
            return pending.template item<int64_t>();
        }

        if (!graph_prepared) {
            prepare_graph();
            graph_prepared = true;
        }
        MfqCudaGuard graph_device_guard(graph.stream.device_index());
        auto graph_stream_guards = activate_cuda_graph_compute_streams(graph.compute_streams);
        cudaStream_t graph_stream = graph.stream.stream();
        if (!sampler.greedy()) {
            *sampler.ops().random_host().template data_ptr<float>() = sampler.next_uniform_float();
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(graph.random.template data_ptr<float>(),
                sampler.ops().random_host().template data_ptr<float>(),
                sizeof(float),
                cudaMemcpyHostToDevice,
                graph_stream));
        }
        graph.graph->replay();
        const auto token = graph.static_next.template item<int64_t>();
        ++state.cache_pos;
        return token;
    }
    void accept(std::int64_t token) {
        if (has_penalties && !graph_active) {
            sample_token_counts_add_cuda(counts, pending.contiguous());
        }
        if constexpr(requires {model.adapter_prefetch_token_rows(model,known_token_host);}) {
            const auto* option=mfq::cuda::env_lookup("TYLOINFER_DECODE_EARLY_PLE");
            if((option && option[0]=='0') || (prepared && prepared->transformed()) ||
               (output && output->stopped()))return;
            if(!known_token_host.defined())known_token_host=mfq_tensor_backend::empty({1,1},
                mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kInt64));
            known_token_host.template data_ptr<int64_t>()[0]=token;
            if(prefetch_token_rows(model,known_token_host)) {
                static bool reported=false;
                if(!reported) {
                    reported=true;
                    std::cerr<<"runtime_decode_early_token_rows enabled=1 host_token_reuse=1\n";
                }
            }
        }
    }
};

} // namespace

template <class Model, class State> struct CudaGenerationContext {
    using Hidden = mfq_tensor_backend::Tensor;
    using Restore = TextSessionRestore;
    Model &model;
    State &state;
    mfq::cuda::Executor* executor;
    CudaExecutionContext &execution;
    const mfq::models::CausalLmMetadata &metadata;
    DecodeGraphCache &graph;
    TextSessionCache &cache;
    const CudaRuntimeConfig &config;
    MtpModule *mtp;
    std::optional<CudaPreparedPrompt> prepared;
    mfq::engine::ContinuousBatch<CudaBatchOperations<Model>>* batching;
    std::string request_id;
    std::function<void()> restore;

    bool batched() const { return batching != nullptr; }
    void trace_expert_cache(const char* boundary, bool drain = true) {
        // Request/first-token boundaries only: existing counters need no
        // events or per-token synchronization. Full summaries drain exchanges.
        static const bool enabled=[] {
            const char* value=mfq::cuda::env_lookup("TYLOINFER_TRACE_MOE_SUMMARY");
            return value && value[0] && value[0]!='0';
        }();
        if (!enabled || batching || std::uncaught_exceptions()) return;
        std::cerr << "moe_request_cache_stats boundary=" << boundary << '\n';
        if(drain)print_moe_expert_cache_stats(execution.stores.moe_expert_cache,std::cerr);
        // The first-token snapshot must not wait for exchanges or introduce
        // CUDA synchronization. Subtract it from the end snapshot to isolate
        // steady decode I/O from the much larger prefill scan.
        std::cerr << "moe_phase_metrics boundary=" << boundary;
        for(const auto& [name,value]:routed_expert_memory_metrics(execution)) {
            if(name=="ssd_expert_read_wait_ms" || name=="expert_route_wait_ms" ||
               name=="expert_host_fetch_ms" || name=="expert_host_copy_ms" || name=="expert_transfer_cache_prepare_ms" || name=="ssd_expert_bytes_read" ||
               name=="ssd_expert_read_count" || name=="ram_expert_cache_hits" ||
               name=="ram_expert_cache_misses" || name=="ram_expert_pcie_read_bytes" ||
               name=="expert_logical_transfer_bytes")std::cerr << ' ' << name << '=' << value;
        }
        std::cerr << '\n';
        const auto* source = model.source.get();
        if(source) {
            const auto io=source->file_read_stats();
            std::cerr<<"runtime_model_io_stats boundary="<<boundary<<" calls="<<io.calls
                <<" logical_bytes="<<io.logical_bytes<<" physical_bytes="<<io.physical_bytes
                <<" read_ms="<<double(io.read_nanoseconds)/1e6<<'\n';
        }
    }
    void prepare_request(const mfq::engine::InferenceRequest& input) {
        trace_expert_cache("begin");
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            model.decode_reserve_until=std::min<int64_t>(metadata.max_position_embeddings,
                int64_t(input.prompt.size())+std::min<int64_t>(input.sampling.max_tokens,256));
            if(!batching && input.prompt.size()>1024) {
                finish_moe_expert_exchanges(execution.stores.moe_expert_cache);
                if(state.cache_pos==0 && !model.graph_registry)
                    release_moe_decode_buffers(execution.stores.moe_expert_cache);
                else release_moe_prefill_buffers(execution.stores.moe_expert_cache);
                mfq::cuda::default_context(mfq_current_cuda_device())->reclaim_idle_memory();
            }
        }
#endif
    }
    void reserve_workspace(std::size_t bytes) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        {
            reserve_moe_workspace(execution.stores.moe_expert_cache,bytes);
            auto context=mfq::cuda::default_context(mfq_current_cuda_device());
            auto available=context->available_memory();
            if(available>=bytes)return;
            std::size_t hot=0;
            for(const auto& metric:cache.metrics())
                if(metric.first=="prefix_cache_hot_bytes")hot=std::size_t(metric.second);
            const auto shortage=bytes-available;
            const auto released=cache.trim_hot(hot>shortage?hot-shortage:0);
            context->reclaim_idle_memory();
            std::cerr<<"runtime_workspace_budget requested_bytes="<<bytes
                <<" available_bytes="<<context->available_memory()
                <<" prefix_released_bytes="<<released<<'\n';
        }
#endif
    }
    bool short_streamed_prefill(std::size_t tokens) const {
        // A short continuation can reuse the bounded MFE path without
        // returning its expert images and captured graphs for a large PP arena.
        return state.cache_pos>0 && tokens<=128 &&
            moe_expert_cache_ssd_experts(execution.stores.moe_expert_cache);
    }
    int64_t prefill_chunk(std::size_t tokens) {
        // Reuse ordinary/MTP widths before allocating wider temporary graphs
        // whose activation storage would displace persistent expert images.
        const auto limit=config.generation.prefill_chunk_size;
        if(short_streamed_prefill(tokens))return std::min<int64_t>(4,limit);
        if(tokens<=128)return std::min<int64_t>(std::max<std::size_t>(1,tokens),limit);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        bool allow_borrow=false;
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>)
            allow_borrow=qwen4_layer_major_prefill_eligible(model,std::min<std::size_t>(tokens,8192)) &&
                (!prepared || !prepared->transformed());
        auto budget=moe_prefill_workspace_budget(execution.stores.moe_expert_cache,allow_borrow);
        const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
        auto available=mfq::engine::prefill_budget_add(
            mfq::engine::prefill_budget_add(context->available_memory(),context->workspace_available()),
            budget.lendable_bytes);
        const auto row_bytes=std::size_t(metadata.hidden_size)*metadata.hc_mult*sizeof(float)*8;
        const auto sequence_bytes=allow_borrow
            ? std::size_t(tokens<=16384?tokens:std::min<std::size_t>(tokens,limit))*row_bytes*3/8 : 0;
        const auto reserve=moe_expert_cache_ssd_experts(execution.stores.moe_expert_cache)
            ? std::max(std::size_t(1)<<30,mfq::cuda::runtime_options::workspace_reserve_bytes())
            : std::size_t(1)<<30;
        const auto fixed=mfq::engine::prefill_budget_add(reserve+sequence_bytes,budget.transfer_bytes);
        const auto wanted=std::min<int64_t>({int64_t(tokens),limit,mfq::engine::max_prefill_compute_chunk});
        const auto required=mfq::engine::prefill_budget_add(fixed,std::size_t(wanted)*row_bytes);
        if(required>available) {
            // Expert overflow in the unused KV budget is revocable too. Ask
            // its existing owner to return pages before reducing the tile.
            reserve_workspace(required>budget.lendable_bytes?required-budget.lendable_bytes:0);
            budget=moe_prefill_workspace_budget(execution.stores.moe_expert_cache,allow_borrow);
            available=mfq::engine::prefill_budget_add(
                context->available_memory()+context->workspace_available(),budget.lendable_bytes);
        }
        const auto selected=mfq::engine::select_prefill_chunk(tokens,limit,available,fixed,row_bytes);
        if(!selected)throw mfq::engine::InferenceInputError(
            mfq::engine::InferenceInputErrorCode::ContextLength,
            "insufficient CUDA workspace for prefill computation and expert transfer buffers");
        std::cerr<<"runtime_prefill_plan mode="<<(config.prefill_chunk_auto?"auto":"limited")
            <<" tokens="<<tokens<<" limit="<<limit<<" compute_chunk="<<selected
            <<" available_bytes="<<available<<" lendable_bytes="<<budget.lendable_bytes
            <<" transfer_bytes="<<budget.transfer_bytes<<'\n';
        return selected;
#else
        return std::min<int64_t>(tokens,limit);
#endif
    }
    void prepare_prefill_workspace(std::size_t tokens) {
        if(short_streamed_prefill(tokens)) {
            std::cerr<<"runtime_streamed_prefill retained_decode=1 tokens="<<tokens
                <<" chunk="<<prefill_chunk(tokens)<<'\n';
            return;
        }
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        const bool streamed=tokens>8 && moe_expert_cache_ssd_experts(execution.stores.moe_expert_cache);
        if constexpr(std::is_same_v<Model,DeepseekV41CausalLm>)
            if(streamed)graph.invalidate();
        const bool retained=streamed && lend_moe_decode_cache_for_prefill(execution.stores.moe_expert_cache,
            mfq::cuda::runtime_options::workspace_reserve_bytes());
        if constexpr(std::is_same_v<Model,DeepseekV41CausalLm>) {
            if(streamed && !retained) {
                // Private FFN graphs retain their transfer arenas after a
                // request. Retire consumers before a new grouped prefill;
                // otherwise these arenas can reduce its ring to one expert.
                release_moe_decode_buffers(execution.stores.moe_expert_cache);
                mfq::cuda::default_context(mfq_current_cuda_device())->reclaim_idle_memory();
            }
        }
#endif
        if(tokens>1024 || moe_expert_cache_ssd_experts(execution.stores.moe_expert_cache))
            reserve_workspace(mfq::cuda::runtime_options::workspace_reserve_bytes());
    }
    bool prepare_snapshot() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if constexpr(std::is_same_v<Model,Qwen4CausalLm> || std::is_same_v<Model,DeepseekV41CausalLm>) {
            // Retire prefill buffers before copying KV/recurrent state. Evict
            // old hot entries before allocating the incoming checkpoint.
            release_moe_prefill_buffers(execution.stores.moe_expert_cache);
            // A prefix stores occupied rows, not the full preallocated context
            // or graph-owned speculative checkpoints.
            std::size_t incoming=0;
            if constexpr(std::is_same_v<Model,Qwen4CausalLm>)incoming=qwen4_prefix_snapshot_bytes(model,mtp);
            else {
                incoming=256; // Rolling Engram history and scalar metadata.
                const auto prefix_bytes=[](const auto& tensor,int64_t rows) -> std::size_t {
                    return tensor.defined()?std::size_t(tensor.size(0))*rows*tensor.size(2)*tensor.element_size():0;
                };
                for(const auto& item:state.blocks) {
                    const auto& state=dynamic_cast<const deepseek_v41_runtime::Block&>(*item).state;
                    incoming+=session_tensor_bytes(state.local_kv)+session_tensor_bytes(state.partial_kv)+
                        session_tensor_bytes(state.partial_score)+prefix_bytes(state.compressed_kv,state.compressed_length)+
                        prefix_bytes(state.index_k,state.compressed_length);
                }
                if(mtp) {
                    const auto& c=model.config;
                    incoming+=std::size_t(c.sliding_window)*sizeof(mfq_half)*
                        (c.n_mtp_layers*c.head_dim+c.hidden*c.dspark_target_layer_ids.size());
                }
            }
            std::size_t hot=0,limit=0;
            for(const auto& metric:cache.metrics()) {
                if(metric.first=="prefix_cache_hot_bytes")hot=std::size_t(metric.second);
                if(metric.first=="prefix_cache_max_bytes")limit=std::size_t(metric.second);
            }
            const auto target=limit>incoming?limit-incoming:0;
            if(hot>target)cache.trim_hot(target);
            // Transposing graph KV can temporarily hold a contiguous copy in
            // addition to the checkpoint. Include that peak and metadata.
            const auto needed=incoming*(std::is_same_v<Model,Qwen4CausalLm>?2:1)+(std::size_t(64)<<20);
            reserve_workspace(needed);
            const auto available=mfq::cuda::default_context(mfq_current_cuda_device())->available_memory();
            if(available<needed) {
                std::cerr<<"runtime_session_cache action=skip reason=workspace_pressure"
                    <<" requested_bytes="<<needed<<" available_bytes="<<available<<'\n';
                return false;
            }
        }
#endif
        return true;
    }
    void invalidate_plan() { graph.invalidate(); }
    std::size_t prefix_boundary=0;
    void set_prefix_boundary(std::size_t tokens) { prefix_boundary=tokens; }
    bool has_hidden(const Hidden &value) const { return value.defined(); }
    void reset() {
        if (batching) return;
        state.reset(1);
        if (mtp)
            mtp->reset(1);
    }
    void release() {
        if (batching) return; // Each row owns its slot cleanup.
        trace_expert_cache("end");
        try { reset(); }
        catch (...) { graph.invalidate(); throw; }
        if (std::uncaught_exceptions()) graph.invalidate();
    }
    Generation plain(mfq::engine::InferenceRequest& input, mfq::engine::InferenceOutput& output,
                     size_t reused, size_t stable) {
        if (batching) return batching->generate(request_id, input, output);
        prepare_prefill_workspace(input.prompt.size()-reused);
        const auto compute_chunk=prefill_chunk(input.prompt.size()-reused);
        auto logical_chunk=compute_chunk;
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            // Bound residual/FFN storage as well as projection tiles. Long
            // prompts advance each bounded chunk through all layers.
            logical_chunk=std::min<int64_t>(logical_chunk,8192);
            if(compute_chunk>8 && reused==0 && (stable==0 || stable==input.prompt.size()) &&
                (!prepared || !prepared->transformed()) &&
                qwen4_layer_major_prefill_eligible(model,input.prompt.size()))
                logical_chunk=input.prompt.size();
        }
        if constexpr(std::is_same_v<Model,DeepseekV41CausalLm>) {
            if(!short_streamed_prefill(input.prompt.size()-reused) &&
                    (!prepared || !prepared->transformed()))logical_chunk=input.prompt.size();
        }
        return mfq::engine::generate_sequence(
            CudaGenerationOps<Model, State>(model, state, executor, execution, metadata, graph, input,
                prepared ? &*prepared : nullptr,
                config.decode_graph,compute_chunk,&output,restore),
            output, input.prompt.size(), reused, stable,logical_chunk);
    }
    Generation speculate(mfq::engine::InferenceRequest &input, mfq::engine::InferenceOutput &output,
        size_t reused, Hidden restored, Hidden *committed) {
        prepare_prefill_workspace(input.prompt.size()-reused);
        return run_mtp_generation(model, state, executor, execution, metadata,
            *mtp,
            input,
            output,
            prefill_chunk(input.prompt.size()-reused),
            prepared ? &*prepared : nullptr,
            reused,
            std::move(restored),
            committed,prefix_boundary,restore);
    }
};

template <typename Model, typename State>
Generation generate(Model &model, State &state, mfq::cuda::Executor* executor,
    CudaExecutionContext &execution,
    const mfq::models::CausalLmMetadata &metadata,
    DecodeGraphCache &graph, TextSessionCache &cache,
    const CudaRuntimeConfig &config, mfq::engine::InferenceRequest &request,
    mfq::engine::InferenceOutput &output, MtpModule *mtp,
    std::optional<CudaPreparedPrompt> prepared,
    mfq::engine::ContinuousBatch<CudaBatchOperations<Model>>* batching, std::string request_id,
    std::function<void()> restore) {
    return mfq::engine::generate_request(
        CudaGenerationContext<Model, State>{model, state, executor, execution, metadata, graph, cache, config, mtp,
                                     std::move(prepared), batching,
                                     std::move(request_id), std::move(restore)},
        request,
        output);
}

#define TYLOINFER_INSTANTIATE_FLOW(MODEL)                                                                \
    template Generation generate(MODEL &, MODEL &, mfq::cuda::Executor*, CudaExecutionContext &,    \
        const mfq::models::CausalLmMetadata &,                                                       \
        DecodeGraphCache &,                                                                        \
        TextSessionCache &,                                                                        \
        const CudaRuntimeConfig &,                                                                 \
        mfq::engine::InferenceRequest &,                                                           \
        mfq::engine::InferenceOutput &,                                                            \
        MtpModule *,                                                                               \
        std::optional<CudaPreparedPrompt>,                                                         \
        mfq::engine::ContinuousBatch<CudaBatchOperations<MODEL>>*, std::string,                    \
        std::function<void()>);
template Generation generate(mfq::cuda::qwen35::Model &, mfq::cuda::qwen35::State &,
    mfq::cuda::Executor*, CudaExecutionContext &,
    const mfq::models::CausalLmMetadata &, DecodeGraphCache &, TextSessionCache &,
    const CudaRuntimeConfig &, mfq::engine::InferenceRequest &, mfq::engine::InferenceOutput &,
    MtpModule *, std::optional<CudaPreparedPrompt>,
    mfq::engine::ContinuousBatch<CudaBatchOperations<mfq::cuda::qwen35::Model>>*, std::string,
    std::function<void()>);
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::MiniCPMO45CausalLm)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::MiniCPMOTtsCausalLm)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::Gemma4CausalLm)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::GlmDsaCausalLm)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::Glm5CausalLm)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::Qwen4CausalLm)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::DeepseekV4CausalLm)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::DeepseekV41CausalLm)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::NemotronHCausalLm)

#undef TYLOINFER_INSTANTIATE_FLOW
} // namespace mfq::cuda::internal
