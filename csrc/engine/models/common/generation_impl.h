#pragma once
#include "engine/weights/workspace.h"

#include "env.h"
#include "engine/weights/routed_cache.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/decode_graph.h"
#include "engine/models/common/causal_metadata.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/requests/output.h"
#include "engine/requests/policy.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/models/common/prefetch.h"
#include "engine/models/deepseek_v41/ops.h"
#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/common/prompt.h"
#include "engine/backends/cuda/executor/sampling.h"
#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/models/input.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <optional>
#include <type_traits>
#include <utility>
namespace mfq::cuda::internal {
namespace generation_detail {

using Tensor = mfq_tensor_backend::Tensor;
using mfq_tensor_backend::indexing::Slice;

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
              Collectives& collectives, const Executor::Mode& mode, mfq::cuda::Input input) {
    static_assert(has_forward<Model, State>);
    TYLOINFER_RUNTIME_CHECK(executor != nullptr,
                            "CUDA model forward requires an executor");
    return forward(model, state, *executor, collectives,
                   mode, std::move(input));
}

template <typename Model>
Tensor project(Model& model, mfq::cuda::Executor* executor,
               const Executor::Mode& mode, Tensor value, bool capped) {
    static_assert(has_logits<Model>);
    TYLOINFER_RUNTIME_CHECK(executor != nullptr,
                            "CUDA model logits requires an executor");
    return logits(model, *executor, std::move(value), capped, mode);
}

template <typename Model, typename State>
Tensor last_logits(Model& model, State& state, mfq::cuda::Executor* executor,
                   Collectives& collectives, const Executor::Mode& mode, Tensor ids, bool capped) {
    auto value = hidden(model, state, executor, collectives, mode,
                        mfq::cuda::Input{std::move(ids)});
    return project(model, executor, mode,
                   value.index({Slice(), -1, Slice()})
                          .to(mfq_tensor_backend::kFloat16)
                          .contiguous(),
                      capped);
}

template <typename Model, typename State>
Tensor sample_token(Model &model, State& state, mfq::cuda::Executor* executor,
    Collectives& collectives, const Executor::Mode& mode, CudaProfiler& profiler, Tensor ids,
    mfq::cuda::Sampler &sampler, mfq_tensor_backend::Tensor counts,
    const MfqTokenConstraintPtr &token_constraint) {
    const auto run = [&]() {
        if (sampler.greedy() && !sampler.has_penalties() && !token_constraint) {
            if constexpr (has_forward<Model, State> && has_logits<Model>) {
                auto logits = last_logits(model, state, executor, collectives, mode,
                                              std::move(ids), false)
                                  .contiguous().view({1, -1});
                return sampler.ops().sample_greedy(std::move(logits));
            } else {
                return model.next_token(ids);
            }
        }
        if constexpr (has_forward<Model, State> && has_logits<Model>) {
            auto logits = last_logits(model, state, executor, collectives, mode,
                                          std::move(ids), true)
                              .contiguous().view({1, -1});
            return mfq::engine::sample_logits(sampler, std::move(logits), counts,
                                            token_constraint);
        } else {
            auto logits = model.last_logits(ids).contiguous().view({1, -1});
            return mfq::engine::sample_logits(sampler, std::move(logits), counts, token_constraint);
        }
    };
    const auto* setting = mfq::cuda::env_lookup("TYLOINFER_PROFILE_DECODE");
    if (!setting || setting[0] != '1' || ids.numel() > 8) return run();
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
    Collectives& collectives;
    const Executor::Mode mode;
    CudaProfiler& profiler;
    RoutedCache& weights;
    bool& serial_branches;
    bool report_memory;
    const mfq::models::CausalLmMetadata &metadata;
    DecodeGraphCache &graph;
    const MfqTokenConstraintPtr &constraint;
    mfq_tensor_backend::Tensor pending, counts;
    mfq_tensor_backend::Tensor known_token_host;
    mfq::cuda::Sampler sampler;
    bool has_penalties;
    const PreparedPrompt *prepared;
    const CudaDecodeGraphConfig graph_config;
    mfq_tensor_backend::Tensor full_ids;
    std::int32_t generation_limit = 0;
    bool graph_active = false;
    bool graph_prepared = false;
    int64_t prefill_compute_chunk=1024;
    mfq::engine::InferenceOutput* output=nullptr;
    std::function<void()> restore;

    CudaGenerationOps(Model &model, State &state, mfq::cuda::Executor* executor,
        Collectives& collectives, Executor::Mode mode, CudaProfiler& profiler,
        RoutedCache& weights, bool& serial_branches, bool report_memory,
        const mfq::models::CausalLmMetadata &metadata, DecodeGraphCache &graph,
        const mfq::engine::InferenceRequest &request, const PreparedPrompt *prepared,
        const CudaDecodeGraphConfig &graph_config,int64_t compute_chunk=1024,
        mfq::engine::InferenceOutput* inference_output=nullptr,
        std::function<void()> restore = {})
        : model(model), state(state), executor(executor), collectives(collectives), mode(mode), profiler(profiler),
          weights(weights), serial_branches(serial_branches), report_memory(report_memory),
          metadata(metadata), graph(graph), constraint(request.token_constraint),
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
        ImageMaskScope image_scope(model,prepared,chunk.offset,chunk.count);
        Tensor hidden;
        bool layer_major=false,short_windows=false;
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            // TQ groups all prompt rows by expert without expanding an entire
            // layer. Splitting these batches into verification windows would
            // repeatedly transfer the same expert when it is not resident.
            const bool verification_windows=!weights.tq || chunk.count<=8;
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
            MoePrefillWorkspaceLoan workspace(weights.moe);
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
                hidden = ::mfq::cuda::internal::generation_detail::hidden(
                    model, state, executor, collectives, mode, std::move(input));
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
                hidden = ::mfq::cuda::internal::generation_detail::hidden(
                    model, state, executor, collectives, mode, mfq::cuda::Input{std::move(ids)});
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
                auto logits = project(model, executor, mode, std::move(input), true)
                                  .reshape({1, -1});
                pending = mfq::engine::sample_logits(sampler, std::move(logits), counts, constraint);
            } else {
                auto logits = model.logits_from_hidden(std::move(input)).reshape({1, -1});
                pending = mfq::engine::sample_logits(sampler, std::move(logits), counts, constraint);
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
               !weights.moe &&
               generation_limit >= graph_config.minimum_generation_tokens &&
               generation_limit <= graph.generated_capacity;
    }
    void prepare_graph() {
        using Tensor = mfq_tensor_backend::Tensor;
        graph.ensure_compute_streams(collectives);
        MfqCudaGuard graph_device_guard(graph.stream.device_index());
        auto graph_stream_guards = activate_cuda_graph_compute_streams(graph.compute_streams);
        if (has_penalties) {
            sample_token_counts_add_cuda(counts, pending.contiguous());
        }

        graph.initialize_inputs(pending, state.cache_pos, sampler.ops().random_host());

        const std::int64_t requested = state.cache_pos + generation_limit;
        const std::int64_t planned =
            decode_graph_bucket(requested, metadata.max_position_embeddings);
        const std::int64_t parts =
            decode_graph_attention_parts(planned, FullAttention::kDecodeAttentionMaxParts);
        const bool greedy = sampler.greedy();
        auto graph_mode = mode;
        graph_mode.ffn.serial = true;
        const auto sample_static = [&]() {
            if constexpr (has_forward<Model, State> && has_logits<Model>) {
                mfq::cuda::Input input;
                input.ids = graph.static_input;
                input.positions = graph.static_pos;
                input.cache_positions = graph.static_pos;
                input.lengths = graph.static_len;
                input.kv_length = planned;
                input.attention_parts = parts;
                auto value = ::mfq::cuda::internal::generation_detail::hidden(
                    model, state, executor, collectives, graph_mode, std::move(input));
                auto logits = project(
                    model, executor, graph_mode,
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
            std::optional<DecodeGraphBranchScope> branch_scope;
            if constexpr (!has_forward<Model, State>) branch_scope.emplace(serial_branches);
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
                    graph.graph_participant_streams(collectives));
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
            report_cuda_memory(report_memory, "runtime_graph_capture");
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
                        pending=greedy?std::move(*result):mfq::engine::sample_logits(sampler,
                            result->contiguous().view({1,-1}),counts,constraint);
                        return pending.template item<int64_t>();
                    }
                }
            }
            pending = sample_token(model, state, executor, collectives, mode, profiler,
                                   pending.reshape({1, 1}), sampler, counts, constraint);
            return pending.template item<int64_t>();
        }

        if (!graph_prepared) {
            prepare_graph();
            graph_prepared = true;
        }
        MfqCudaGuard graph_device_guard(graph.stream.device_index());
        auto graph_stream_guards = activate_cuda_graph_compute_streams(graph.compute_streams);
        if (!sampler.greedy()) {
            *sampler.ops().random_host().template data_ptr<float>() = sampler.next_uniform_float();
            graph.upload_random(sampler.ops().random_host());
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

} // namespace generation_detail
} // namespace mfq::cuda::internal
