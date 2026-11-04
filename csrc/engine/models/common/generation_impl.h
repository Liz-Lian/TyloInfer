#pragma once
#include "engine/weights/workspace.h"

#include "env.h"
#include "engine/weights/routed_cache.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/decode_graph.h"
#include "engine/backends/cuda/ops/attention/scratch.h"
#include "engine/models/info.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/requests/output.h"
#include "engine/requests/policy.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/models/common/prefetch.h"
#include "engine/models/common/generation_ops.h"
#include "engine/models/common/prompt.h"
#include "engine/backends/cuda/executor/sampling.h"
#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/models/common/input.h"
#include "engine/models/causal_model.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <optional>
#include <utility>
namespace tyloi::cuda::internal {
namespace generation_detail {

using DenseTensor = ::tyloi::DenseTensor;
using ::tyloi::indexing::Slice;

template <typename Model, typename State>
constexpr bool has_forward =
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    requires(Model& model, State& state, tyloi::cuda::Input input,
             const tyloi::models::ForwardOptions& options) {
        static_cast<const tyloi::models::CausalModel&>(model).forward(std::move(input), state, options);
    } ||
#endif
    requires(
    Model& model, State& state, tyloi::cuda::Executor& executor,
    tyloi::cuda::Collectives& collectives, const tyloi::cuda::Executor::Mode& mode,
    tyloi::cuda::Input input) {
    forward(model, state, executor, collectives, mode, std::move(input));
};

template <typename Model>
constexpr bool has_logits =
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    requires(Model& model, DenseTensor hidden) {
        static_cast<const tyloi::models::CausalModel&>(model).logits(std::move(hidden), true);
    } ||
#endif
    requires(
    Model& model, tyloi::cuda::Executor& executor, DenseTensor hidden) {
    logits(model, executor, std::move(hidden), true);
};

template <typename Model, typename State>
DenseTensor hidden(Model& model, State& state, tyloi::cuda::Executor* executor,
              Collectives& collectives, const Executor::Mode& mode, tyloi::cuda::Input input) {
    static_assert(has_forward<Model, State>);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    if constexpr (requires { static_cast<const tyloi::models::CausalModel&>(model).forward(
                               std::move(input), state, forward_options(mode)); }) {
        const tyloi::models::CausalModel& causal = model;
        auto scope = Executor::current().enter(mode.ffn);
        return DenseTensor(causal.forward(std::move(input), state, forward_options(mode)));
    } else
#endif
    {
        TYLOINFER_RUNTIME_CHECK(executor != nullptr,
                                "CUDA model forward requires an executor");
        return forward(model, state, *executor, collectives,
                       mode, std::move(input));
    }
}

template <typename Model>
DenseTensor project(Model& model, tyloi::cuda::Executor* executor,
               const Executor::Mode& mode, DenseTensor value, bool capped) {
    static_assert(has_logits<Model>);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    if constexpr (requires { static_cast<const tyloi::models::CausalModel&>(model).logits(
                               std::move(value), capped); }) {
        const tyloi::models::CausalModel& causal = model;
        auto scope = Executor::current().enter(mode.ffn);
        return DenseTensor(causal.logits(std::move(value), capped));
    } else
#endif
    {
        TYLOINFER_RUNTIME_CHECK(executor != nullptr,
                                "CUDA model logits requires an executor");
        return logits(model, *executor, std::move(value), capped, mode);
    }
}

template <typename Model, typename State>
DenseTensor last_logits(Model& model, State& state, tyloi::cuda::Executor* executor,
                   Collectives& collectives, const Executor::Mode& mode, DenseTensor ids, bool capped) {
    auto value = hidden(model, state, executor, collectives, mode,
                        tyloi::cuda::Input{std::move(ids)});
    return project(model, executor, mode,
                   value.index({Slice(), -1, Slice()})
                          .to(::tyloi::kFloat16)
                          .contiguous(),
                      capped);
}

template <typename Model, typename State>
DenseTensor sample_token(Model &model, State& state, tyloi::cuda::Executor* executor,
    Collectives& collectives, const Executor::Mode& mode, CudaProfiler& profiler, DenseTensor ids,
    tyloi::cuda::Sampler &sampler, ::tyloi::DenseTensor counts,
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
            return tyloi::engine::sample_logits(sampler, std::move(logits), counts,
                                            token_constraint);
        } else {
            auto logits = model.last_logits(ids).contiguous().view({1, -1});
            return tyloi::engine::sample_logits(sampler, std::move(logits), counts, token_constraint);
        }
    };
    const auto* setting = tyloi::cuda::env_lookup("TYLOINFER_PROFILE_DECODE");
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
    tyloi::cuda::Executor* executor;
    Collectives& collectives;
    const Executor::Mode mode;
    CudaProfiler& profiler;
    RoutedCache& weights;
    bool& serial_branches;
    bool report_memory;
    const tyloi::models::Info &info;
    DecodeGraphCache &graph;
    const MfqTokenConstraintPtr &constraint;
    ::tyloi::DenseTensor pending, counts;
    ::tyloi::DenseTensor known_token_host;
    tyloi::cuda::Sampler sampler;
    bool has_penalties;
    const PreparedPrompt *prepared;
    const CudaDecodeGraphConfig graph_config;
    ::tyloi::DenseTensor full_ids;
    std::int32_t generation_limit = 0;
    bool graph_active = false;
    bool graph_prepared = false;
    int64_t prefill_compute_chunk=1024;
    tyloi::engine::InferenceOutput* output=nullptr;
    std::function<void()> restore;

    CudaGenerationOps(Model &model, State &state, tyloi::cuda::Executor* executor,
        Collectives& collectives, Executor::Mode mode, CudaProfiler& profiler,
        RoutedCache& weights, bool& serial_branches, bool report_memory,
        const tyloi::models::Info &info, DecodeGraphCache &graph,
        const tyloi::engine::InferenceRequest &request, const PreparedPrompt *prepared,
        const CudaDecodeGraphConfig &graph_config,int64_t compute_chunk=1024,
        tyloi::engine::InferenceOutput* inference_output=nullptr,
        std::function<void()> restore = {})
        : model(model), state(state), executor(executor), collectives(collectives), mode(mode), profiler(profiler),
          weights(weights), serial_branches(serial_branches), report_memory(report_memory),
          info(info), graph(graph), constraint(request.token_constraint),
          sampler(request.sampling),
          has_penalties(sampler.has_penalties()), prepared(prepared), graph_config(graph_config),
          prefill_compute_chunk(compute_chunk),output(inference_output),restore(std::move(restore)) {
        full_ids = ::tyloi::cuda::tensor(request.prompt,
            ::tyloi::TensorOptions()
                .dtype(::tyloi::kInt64)
                .device(::tyloi::kCUDA))
                       .reshape({1, -1})
                       .contiguous();
        graph.ensure_storage(info.vocab_size);
        if (has_penalties) {
            counts = tyloi::cuda::SamplingOps::token_counts(full_ids, info.vocab_size, graph.counts);
        }
        generation_limit = request.sampling.max_tokens;
        graph_active = graph_eligible();
    }

    double prefill(const tyloi::engine::PrefillChunk &chunk) {
        using DenseTensor = ::tyloi::DenseTensor;
        PrefillCudaTimer timer;
        bool prefill_finished=false;
        const auto finish_prefill=[&] {
            if(prefill_finished)return;
            TYLOINFER_CUDA_CHECK(cudaEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
            prefill_finished=true;
        };
        auto ids = full_ids.narrow(1, chunk.offset, chunk.count).contiguous();
        ImageMaskScope image_scope(model,prepared,chunk.offset,chunk.count);
        auto [hidden,layer_major,short_windows]=generation_prefill(model,*this,chunk,ids,finish_prefill);
        if(output && output->stopped()) {
            finish_prefill();
            return timer.elapsed_ms();
        }
        if (hidden.defined()) {
            // The layer-major path has already advanced every physical cache.
        } else if (prepared && prepared->transformed()) {
            WeightLoan workspace(weights.moe);
            const bool borrowed=generation_workspace(model,workspace,chunk.count);
            if constexpr (has_forward<Model, State>) {
                tyloi::cuda::Input input;
                input.ids = ids;
                input.embeddings = DenseTensor(prepared->embeddings).narrow(1, chunk.offset, chunk.count)
                                        .contiguous();
                if (prepared->positions.defined())
                    input.positions = DenseTensor(prepared->positions).narrow(-1, chunk.offset, chunk.count)
                                           .contiguous();
                input.advance = true;
                hidden = ::tyloi::cuda::internal::generation_detail::hidden(
                    model, state, executor, collectives, mode, std::move(input));
            } else {
                hidden = model.hidden_forward_inputs(ids,
                    DenseTensor(prepared->embeddings).narrow(1, chunk.offset, chunk.count).contiguous(),
                    prepared->positions.defined()?MfqOptional<DenseTensor>(
                        DenseTensor(prepared->positions).narrow(-1,chunk.offset,chunk.count).contiguous()):mfq_nullopt,
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
            state.set_position_delta(prepared->decode_position_delta);
        } else {
            if constexpr (has_forward<Model, State>) {
                hidden = ::tyloi::cuda::internal::generation_detail::hidden(
                    model, state, executor, collectives, mode, tyloi::cuda::Input{std::move(ids)});
            } else {
                MfqOptional<DenseTensor> length = mfq_nullopt;
                if (model.adapter_uses_decode_sequence_length() && chunk.offset > 0 && chunk.count == 1)
                    length = ::tyloi::cuda::full({1}, chunk.offset + 1, full_ids.options());
                hidden = model.hidden_forward(ids, mfq_nullopt, length);
            }
        }
        // Match prompt-progress timing: first-token head/sampling and expert
        // restoration remain in TTFT, outside reported prefill throughput.
        finish_prefill();
        if (chunk.offset + chunk.count == full_ids.size(1)) {
            auto input = hidden.index({Slice(), -1, Slice()})
                             .to(::tyloi::kFloat16)
                             .contiguous();
            if constexpr (has_logits<Model>) {
                auto logits = project(model, executor, mode, std::move(input), true)
                                  .reshape({1, -1});
                pending = tyloi::engine::sample_logits(sampler, std::move(logits), counts, constraint);
            } else {
                auto logits = model.logits_from_hidden(std::move(input)).reshape({1, -1});
                pending = tyloi::engine::sample_logits(sampler, std::move(logits), counts, constraint);
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
               info.decode_graph && mfq_cuda_graph_capture_supported() &&
               !weights.moe &&
               generation_limit >= graph_config.minimum_generation_tokens &&
               generation_limit <= graph.generated_capacity;
    }
    void prepare_graph() {
        using DenseTensor = ::tyloi::DenseTensor;
        graph.ensure_compute_streams(collectives);
        MfqCudaGuard graph_device_guard(graph.stream.device_index());
        auto graph_stream_guards = activate_cuda_graph_compute_streams(graph.compute_streams);
        if (has_penalties) {
            sample_token_counts_add_cuda(counts, pending.contiguous());
        }

        graph.initialize_inputs(pending, state.position(), sampler.ops().random_host());

        const std::int64_t requested = state.position() + generation_limit;
        const std::int64_t planned =
            decode_graph_bucket(requested, info.max_context);
        const std::int64_t parts =
            decode_graph_attention_parts(planned, AttentionScratch::kMaxParts);
        const bool greedy = sampler.greedy();
        auto graph_mode = mode;
        graph_mode.ffn.serial = true;
        const auto sample_static = [&]() {
            if constexpr (has_forward<Model, State> && has_logits<Model>) {
                tyloi::cuda::Input input;
                input.ids = graph.static_input;
                input.positions = graph.static_pos;
                input.cache_positions = graph.static_pos;
                input.lengths = graph.static_len;
                input.kv_length = planned;
                input.attention_parts = parts;
                auto value = ::tyloi::cuda::internal::generation_detail::hidden(
                    model, state, executor, collectives, graph_mode, std::move(input));
                auto logits = project(
                    model, executor, graph_mode,
                    value.index({Slice(), -1, Slice()})
                        .to(::tyloi::kFloat16)
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
            std::vector<DenseTensor> result;
            const auto append = [&result](const auto& tensor) {
                if (tensor.defined()) result.emplace_back(tensor);
            };
            if constexpr (requires(const State& value) { value.storage(); }) {
                for (const auto& tensor : state.storage()) append(tensor);
            }
            if (executor) {
                for (const auto& tensor : executor->storage()) append(tensor);
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
                const bool double_warmup = info.double_warmup;
                auto warmup_states = state.graph_state();
                prepare_decode_graph_memory(warmup_states, state,
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
            if(auto result=generation_decode(model,*this))return *result;
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
        state.advance(1);
        return token;
    }
    void accept(std::int64_t token) {
        if (has_penalties && !graph_active) {
            sample_token_counts_add_cuda(counts, pending.contiguous());
        }
        if constexpr(requires {model.adapter_prefetch_token_rows(model,known_token_host);}) {
            const auto* option=tyloi::cuda::env_lookup("TYLOINFER_DECODE_EARLY_PLE");
            if((option && option[0]=='0') || (prepared && prepared->transformed()) ||
               (output && output->stopped()))return;
            if(!known_token_host.defined())known_token_host=::tyloi::cuda::empty({1,1},
                ::tyloi::TensorOptions().device(::tyloi::kCPU).dtype(::tyloi::kInt64));
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
} // namespace tyloi::cuda::internal
