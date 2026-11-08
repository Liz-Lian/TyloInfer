#pragma once

#include "engine/backends/cuda/native/env.h"
#include "engine/generation/cuda.h"
#include "engine/budget.h"
#include "engine/weights/cache.h"
#include "engine/weights/workspace.h"
#include "engine/config.h"

#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/decode_graph.h"
#include "engine/backends/cuda/ops/attention/scratch.h"
#include "engine/models/metadata.h"
#include "engine/models/source.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/requests/output.h"
#include "engine/requests/policy.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/generation/ops.h"
#include "engine/models/common/prompt.h"
#include "engine/backends/cuda/executor/sampling.h"
#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"
#include "engine/models/input.h"
#include "engine/models/causal_model.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <type_traits>
#include <utility>
namespace tyloi::cuda::internal {
namespace generation_detail {

using DenseTensor = ::tyloi::DenseTensor;
using ::tyloi::indexing::Slice;

template <typename Model>
constexpr bool causal =
#ifdef TYLOINFER_NATIVE_CUDA
    std::is_base_of_v<tyloi::models::CausalModel, Model>;
#else
    false;
#endif

template <typename Model, typename State>
constexpr bool has_forward =
    causal<Model> ||
    requires(
    Model& model, State& state, tyloi::cuda::Executor& executor,
    tyloi::cuda::Collectives& collectives, const tyloi::cuda::Executor::Mode& mode,
    tyloi::models::Input input) {
    forward(model, state, executor, collectives, mode, std::move(input));
};

template <typename Model>
constexpr bool has_logits =
    causal<Model> ||
    requires(
    Model& model, tyloi::cuda::Executor& executor,
#ifdef TYLOINFER_NATIVE_CUDA
    const ::tyloi::Tensor& hidden) {
#else
    DenseTensor hidden) {
#endif
    logits(model, executor, std::move(hidden), true);
};

template <typename Model, typename State>
auto hidden(Model& model, State& state, tyloi::cuda::Executor* executor,
              Collectives& collectives, const Executor::Mode& mode, tyloi::models::Input input) {
    static_assert(has_forward<Model, State>);
    TYLOINFER_RUNTIME_CHECK(executor != nullptr,
                            "CUDA model forward requires an executor");
#ifdef TYLOINFER_NATIVE_CUDA
    auto scope = Executor::current().enter(mode.ffn);
    if constexpr (causal<Model>) {
        return model.forward(std::move(input), state, forward_options(mode));
    } else {
        return forward(model, state, *executor, collectives, mode, std::move(input));
    }
#else
    return forward(model, state, *executor, collectives, mode, std::move(input));
#endif
}

template <typename Model, typename Value>
auto project(Model& model, tyloi::cuda::Executor* executor,
               const Executor::Mode& mode, Value value, bool capped) {
    static_assert(has_logits<Model>);
    TYLOINFER_RUNTIME_CHECK(executor != nullptr,
                            "CUDA model logits requires an executor");
#ifdef TYLOINFER_NATIVE_CUDA
    auto scope = Executor::current().enter(mode.ffn);
    if constexpr (causal<Model>) {
        return model.logits(value, capped);
    } else {
        return logits(model, *executor, value, capped, mode);
    }
#else
    return logits(model, *executor, std::move(value), capped, mode);
#endif
}

template <typename Model, typename State, typename Value>
auto last_logits(Model& model, State& state, tyloi::cuda::Executor* executor,
                   Collectives& collectives, const Executor::Mode& mode, Value ids, bool capped) {
    auto value = hidden(model, state, executor, collectives, mode,
                        tyloi::models::Input{std::move(ids)});
#ifdef TYLOINFER_NATIVE_CUDA
    return project(model, executor, mode,
                   tyloi::ops::contiguous(tyloi::ops::cast(tyloi::ops::select(value, 1, -1),
                                                            ::tyloi::kFloat16)), capped);
#else
    return project(model, executor, mode,
                   value.index({Slice(), -1, Slice()}).to(::tyloi::kFloat16).contiguous(), capped);
#endif
}

template <typename Model, typename State, typename Value>
auto sample_token(Model &model, State& state, tyloi::cuda::Executor* executor,
    Collectives& collectives, const Executor::Mode& mode, CudaProfiler& profiler, Value ids,
    tyloi::cuda::Sampler &sampler, Value counts,
    const MfqTokenConstraintPtr &token_constraint) {
    const auto run = [&]() {
        if (sampler.greedy() && !sampler.has_penalties() && !token_constraint) {
#ifdef TYLOINFER_NATIVE_CUDA
            auto logits = tyloi::ops::reshape(tyloi::ops::contiguous(last_logits(
                model, state, executor, collectives, mode, std::move(ids), false)), {1, -1});
            return tyloi::ops::sample_greedy(logits);
#else
            auto logits = last_logits(model, state, executor, collectives, mode,
                                      std::move(ids), false).contiguous().view({1, -1});
            return sampler.ops().sample_greedy(std::move(logits));
#endif
        }
#ifdef TYLOINFER_NATIVE_CUDA
        auto logits = tyloi::ops::reshape(tyloi::ops::contiguous(last_logits(
            model, state, executor, collectives, mode, std::move(ids), true)), {1, -1});
#else
        auto logits = last_logits(model, state, executor, collectives, mode,
                                  std::move(ids), true).contiguous().view({1, -1});
#endif
        return tyloi::engine::sample_logits(sampler, std::move(logits), counts, token_constraint);
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
    ::tyloi::weights::Cache& weights;
    bool& serial_branches;
    bool report_memory;
    const tyloi::models::Info &info;
    DecodeGraphCache &graph;
    const MfqTokenConstraintPtr &constraint;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor pending, counts, known_token_host;
#else
    ::tyloi::DenseTensor pending, counts, known_token_host;
#endif
    tyloi::cuda::Sampler sampler;
    bool has_penalties;
    const tyloi::models::Prepared *prepared;
    const tyloi::engine::GraphConfig graph_config;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor full_ids;
#else
    ::tyloi::DenseTensor full_ids;
#endif
    std::int32_t generation_limit = 0;
    bool graph_active = false;
    bool graph_prepared = false;
    int64_t prefill_compute_chunk=1024;
    tyloi::engine::InferenceOutput* output=nullptr;
    tyloi::engine::Budget* budget=nullptr;
    const tyloi::ModelSource* source=nullptr;

    CudaGenerationOps(Model &model, State &state, tyloi::cuda::Executor* executor,
        Collectives& collectives, Executor::Mode mode, CudaProfiler& profiler,
        ::tyloi::weights::Cache& weights, bool& serial_branches, bool report_memory,
        const tyloi::models::Info &info, DecodeGraphCache &graph,
        const tyloi::engine::InferenceRequest &request, const tyloi::models::Prepared *prepared,
        const tyloi::engine::GraphConfig &graph_config,int64_t compute_chunk=1024,
        tyloi::engine::InferenceOutput* inference_output=nullptr,
        tyloi::engine::Budget* budget=nullptr, const tyloi::ModelSource* source=nullptr)
        : model(model), state(state), executor(executor), collectives(collectives), mode(mode), profiler(profiler),
          weights(weights), serial_branches(serial_branches), report_memory(report_memory),
          info(info), graph(graph), constraint(request.token_constraint),
#ifdef TYLOINFER_NATIVE_CUDA
          sampler(request.sampling, tyloi::Device(tyloi::kCUDA, mode.primary)),
#else
          sampler(request.sampling),
#endif
          has_penalties(sampler.has_penalties()), prepared(prepared), graph_config(graph_config),
          prefill_compute_chunk(compute_chunk),output(inference_output),budget(budget),source(source) {
#ifdef TYLOINFER_NATIVE_CUDA
        full_ids = tyloi::ops::contiguous(tyloi::ops::reshape(
            tyloi::ops::tensor(request.prompt, tyloi::Device(tyloi::kCUDA, mode.primary)), {1, -1}));
#else
        full_ids = ::tyloi::cuda::tensor(request.prompt,
            ::tyloi::TensorOptions()
                .dtype(::tyloi::kInt64)
                .device(::tyloi::kCUDA))
                       .reshape({1, -1})
                       .contiguous();
#endif
        graph.ensure_storage(info.vocab_size);
        if (has_penalties) {
#ifdef TYLOINFER_NATIVE_CUDA
            counts = graph.counts;
            tyloi::ops::fill(counts, 0);
            tyloi::ops::add_token_counts(counts, full_ids);
#else
            counts = tyloi::cuda::SamplingOps::token_counts(full_ids, info.vocab_size, graph.counts);
#endif
        }
        generation_limit = request.sampling.max_tokens;
        graph_active = graph_eligible();
    }

    double prefill(const tyloi::engine::PrefillChunk &chunk) {
#ifdef TYLOINFER_NATIVE_CUDA
        auto scope = Executor::current().enter(mode.ffn);
#endif
        PrefillCudaTimer timer;
        bool prefill_finished=false;
        const auto finish_prefill=[&] {
            if(prefill_finished)return;
            TYLOINFER_CUDA_CHECK(cudaEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
            prefill_finished=true;
        };
#ifdef TYLOINFER_NATIVE_CUDA
        auto ids = tyloi::ops::contiguous(tyloi::ops::slice(full_ids, 1, chunk.offset, chunk.offset + chunk.count));
#else
        auto ids = full_ids.narrow(1, chunk.offset, chunk.count).contiguous();
#endif
        ImageMaskScope image_scope(state,prepared,chunk.offset,chunk.count);
        auto [hidden,layer_major,short_windows]=generation_prefill(model,*this,chunk,ids,finish_prefill);
        if(output && output->stopped()) {
            finish_prefill();
            return timer.elapsed_ms();
        }
        if (hidden.defined()) {
            // The layer-major path has already advanced every physical cache.
        } else if (prepared && prepared->transformed()) {
            tyloi::weights::Loan workspace(weights);
            const bool borrowed=generation_workspace(model,workspace,chunk.count);
            tyloi::models::Input input;
#ifdef TYLOINFER_NATIVE_CUDA
            input.ids = ids;
            if (prepared->embeddings)
                input.embeddings = tyloi::ops::contiguous(tyloi::ops::slice(prepared->embeddings, 1,
                    chunk.offset, chunk.offset + chunk.count));
            if (prepared->positions)
                input.positions = tyloi::ops::contiguous(tyloi::ops::slice(prepared->positions, -1,
                    chunk.offset, chunk.offset + chunk.count));
#else
            input.ids = ids;
            input.embeddings = prepared->embeddings.dense().narrow(1, chunk.offset, chunk.count)
                                    .contiguous();
            if (prepared->positions.defined())
                input.positions = prepared->positions.dense().narrow(-1, chunk.offset, chunk.count)
                                       .contiguous();
#endif
            input.advance = true;
            hidden = ::tyloi::cuda::internal::generation_detail::hidden(
                model, state, executor, collectives, mode, std::move(input));
            finish_prefill();
            if(borrowed) {
                workspace.suspend();
#ifdef TYLOINFER_NATIVE_CUDA
                hidden=tyloi::ops::clone(tyloi::ops::slice(hidden,1,hidden.size(1)-1,hidden.size(1)));
#else
                hidden=hidden.narrow(1,hidden.size(1)-1,1).clone();
#endif
                workspace.finish();
            }
            state.set_position_delta(prepared->decode_position_delta);
        } else {
            hidden = ::tyloi::cuda::internal::generation_detail::hidden(
                model, state, executor, collectives, mode, tyloi::models::Input{std::move(ids)});
        }
        // Match prompt-progress timing: first-token head/sampling and expert
        // restoration remain in TTFT, outside reported prefill throughput.
        finish_prefill();
        if (chunk.offset + chunk.count == full_ids.size(1)) {
#ifdef TYLOINFER_NATIVE_CUDA
            auto input = tyloi::ops::contiguous(tyloi::ops::cast(tyloi::ops::select(hidden, 1, -1),
                                                               ::tyloi::kFloat16));
#else
            auto input = hidden.index({Slice(), -1, Slice()}).to(::tyloi::kFloat16).contiguous();
#endif
#ifdef TYLOINFER_NATIVE_CUDA
            auto logits = tyloi::ops::reshape(project(model, executor, mode, std::move(input), true), {1, -1});
#else
            auto logits = project(model, executor, mode, std::move(input), true).reshape({1, -1});
#endif
            pending = tyloi::engine::sample_logits(sampler, std::move(logits), counts, constraint);
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
#ifdef TYLOINFER_NATIVE_CUDA
        const auto token=tyloi::ops::item(pending);
#else
        const auto token=pending.template item<int64_t>();
#endif
        // Prefill temporaries have retired; expert-cache maintenance remains
        // before token publication, inside TTFT for every CUDA model family.
        if (budget) weights.prepare(::tyloi::weights::Cache::Stage::decode, tyloi::Device(tyloi::kCUDA, mode.primary), source);
        return token;
    }

    bool graph_eligible() const {
        return graph_config.enabled && (!prepared || !prepared->transformed()) && !constraint &&
               info.decode_graph && mfq_cuda_graph_capture_supported() &&
               weights.graph_safe(::tyloi::weights::Cache::Graph::legacy) &&
               generation_limit >= graph_config.minimum_generation_tokens &&
               generation_limit <= graph.generated_capacity;
    }
    void prepare_graph() {
        using DenseTensor = ::tyloi::DenseTensor;
        graph.ensure_compute_streams(collectives);
        MfqCudaGuard graph_device_guard(graph.stream.device_index());
        auto graph_stream_guards = activate_cuda_graph_compute_streams(graph.compute_streams);
#ifdef TYLOINFER_NATIVE_CUDA
        if (has_penalties) tyloi::ops::add_token_counts(counts, tyloi::ops::contiguous(pending));
        graph.initialize_inputs(pending, state.position(), sampler.random_host());
#else
        if (has_penalties) sample_token_counts_add_cuda(counts, pending.contiguous());
        auto random_host = sampler.ops().random_host();
        graph.initialize_inputs(pending, state.position(), random_host);
#endif

        const std::int64_t requested = state.position() + generation_limit;
        const std::int64_t planned =
            decode_graph_bucket(requested, info.max_context);
        const std::int64_t parts =
            decode_graph_attention_parts(planned, AttentionScratch::kMaxParts);
        const bool greedy = sampler.greedy();
        auto graph_mode = mode;
        graph_mode.ffn.serial = true;
        const auto sample_static = [&]() {
            tyloi::models::Input input;
            input.ids = graph.static_input;
            input.positions = graph.static_pos;
            input.cache_positions = graph.static_pos;
            input.lengths = graph.static_len;
            input.kv_length = planned;
            input.attention_parts = parts;
            auto value = ::tyloi::cuda::internal::generation_detail::hidden(
                model, state, executor, collectives, graph_mode, std::move(input));
#ifdef TYLOINFER_NATIVE_CUDA
            auto logits = tyloi::ops::reshape(tyloi::ops::contiguous(project(
                model, executor, graph_mode,
                tyloi::ops::contiguous(tyloi::ops::cast(tyloi::ops::select(value, 1, -1), ::tyloi::kFloat16)),
                !(greedy && !has_penalties))), {1, -1});
            if (has_penalties) sampler.apply_penalties(logits, counts);
            if (greedy) return tyloi::ops::sample_greedy(logits);
            return tyloi::ops::sample_stochastic(logits, graph.random, sampler.params().temperature,
                                                 sampler.params().top_k, sampler.params().top_p);
#else
            auto logits = project(
                model, executor, graph_mode,
                value.index({Slice(), -1, Slice()}).to(::tyloi::kFloat16).contiguous(),
                !(greedy && !has_penalties)).contiguous().view({1, -1});
            if (has_penalties) logits = sampler.apply_penalties(std::move(logits), counts);
            if (greedy) return sampler.ops().sample_greedy(std::move(logits));
            return sampler.ops().sample_stochastic(std::move(logits), graph.random, sampler.params());
#endif
        };
        const auto current_graph_storage = [&]() {
#ifdef TYLOINFER_NATIVE_CUDA
            std::vector<std::shared_ptr<const tyloi::Tensor>> result;
            const auto append = [&result](const auto& tensor) {
                if (tensor.defined()) result.emplace_back(tensor);
            };
#else
            std::vector<DenseTensor> result;
            const auto append = [&result](const auto& tensor) {
                if (tensor.defined()) result.emplace_back(tensor);
            };
#endif
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
            if constexpr (causal<Model>) branch_scope.emplace(serial_branches);
            graph.graph = std::make_unique<MfqCudaGraph>();
            try {
                const bool double_warmup = info.double_warmup;
                auto warmup_states = state.graph_state();
                prepare_decode_graph_memory(warmup_states, state,
                    double_warmup, *graph.graph, sample_static,
                    graph.graph_participant_streams(collectives));
                graph.graph->capture_begin();
#ifdef TYLOINFER_NATIVE_CUDA
                auto next = sample_static();
                if (has_penalties) tyloi::ops::add_token_counts(counts, tyloi::ops::contiguous(next));
                graph.capture_next(std::move(next));
#else
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
#endif
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
#ifdef TYLOINFER_NATIVE_CUDA
            pending = sample_token(model, state, executor, collectives, mode, profiler,
                                   tyloi::ops::reshape(pending, {1, 1}), sampler, counts, constraint);
            return tyloi::ops::item(pending);
#else
            pending = sample_token(model, state, executor, collectives, mode, profiler,
                                   pending.reshape({1, 1}), sampler, counts, constraint);
            return pending.template item<int64_t>();
#endif
        }

        if (!graph_prepared) {
            prepare_graph();
            graph_prepared = true;
        }
        MfqCudaGuard graph_device_guard(graph.stream.device_index());
        auto graph_stream_guards = activate_cuda_graph_compute_streams(graph.compute_streams);
        if (!sampler.greedy()) {
#ifdef TYLOINFER_NATIVE_CUDA
            graph.upload_random(sampler.next_random());
#else
            *sampler.ops().random_host().template data_ptr<float>() = sampler.next_uniform_float();
            graph.upload_random(sampler.ops().random_host());
#endif
        }
        graph.graph->replay();
#ifdef TYLOINFER_NATIVE_CUDA
        const auto token = tyloi::ops::item(*graph.static_next);
#else
        const auto token = graph.static_next.template item<int64_t>();
#endif
        state.advance(1);
        return token;
    }
    void accept(std::int64_t token) {
        if (has_penalties && !graph_active) {
#ifdef TYLOINFER_NATIVE_CUDA
            tyloi::ops::add_token_counts(counts, tyloi::ops::contiguous(pending));
#else
            sample_token_counts_add_cuda(counts, pending.contiguous());
#endif
        }
        if constexpr(requires {model.prefetch_token_rows(state,known_token_host);}) {
            const auto* option=tyloi::cuda::env_lookup("TYLOINFER_DECODE_EARLY_PLE");
            if((option && option[0]=='0') || (prepared && prepared->transformed()) ||
               (output && output->stopped()))return;
#ifdef TYLOINFER_NATIVE_CUDA
            if(!known_token_host.defined())known_token_host=tyloi::ops::empty({1,1},
                ::tyloi::TensorOptions().device(::tyloi::kCPU).dtype(::tyloi::kInt64));
            tyloi::ops::fill(known_token_host, token);
#else
            if(!known_token_host.defined())known_token_host=::tyloi::cuda::empty({1,1},
                ::tyloi::TensorOptions().device(::tyloi::kCPU).dtype(::tyloi::kInt64));
            known_token_host.template data_ptr<int64_t>()[0]=token;
#endif
            if(model.prefetch_token_rows(state,known_token_host)) {
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
#include "engine/generation/generation.h"
#include "engine/models/session/cache.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>

namespace tyloi::cuda::internal {
using tyloi::engine::Generation;
namespace generation_detail {

template <class Model, class State, class Predictor> struct RequestOps {
    using Hidden = std::shared_ptr<tyloi::Tensor>;
    using Restore = TextSessionRestore;
    Model &model;
    State &state;
    tyloi::cuda::Executor* executor;
    tyloi::cuda::Collectives& collectives;
    const Executor::Mode mode;
    CudaProfiler& profiler;
    ::tyloi::weights::Cache& weights;
    tyloi::engine::Budget& budget;
    bool& serial_branches;
    bool report_memory;
    bool graph_allowed;
    std::function<void()> drain;
    const tyloi::models::Info &info;
    const tyloi::ModelSource* source;
    DecodeGraphCache &graph;
    tyloi::engine::SessionCache<CudaSessionOps> &cache;
    const tyloi::engine::Config &config;
    tyloi::engine::mtp::State &mtp_state;
    Predictor *mtp;
    std::optional<tyloi::models::Prepared> prepared;
    tyloi::engine::Batch* batching;
    std::string request_id;
    bool decode_ready;

    bool batched() const { return batching != nullptr; }
    void trace_expert_cache(const char* boundary, bool wait = true) {
        // Request/first-token boundaries only: existing counters need no
        // events or per-token synchronization. Full summaries drain exchanges.
        static const bool enabled=[] {
            const char* value=tyloi::cuda::env_lookup("TYLOINFER_TRACE_MOE_SUMMARY");
            return value && value[0] && value[0]!='0';
        }();
        if (!enabled || batching || std::uncaught_exceptions()) return;
        weights.trace(boundary, wait, source);
    }
    void prepare_request(const tyloi::engine::InferenceRequest& input) {
        trace_expert_cache("begin");
#ifdef TYLOINFER_NATIVE_CUDA
        generation_request(model,*this,input);
#endif
    }
    int64_t prefill_chunk(std::size_t tokens) {
        // Reuse ordinary/MTP widths before allocating wider temporary graphs
        // whose activation storage would displace persistent expert images.
        const auto limit=config.generation.prefill_chunk_size;
        if(weights.prefill_mode(tokens, state.position() > 0) == ::tyloi::weights::Cache::Prefill::retained)
            return std::min<int64_t>(4,limit);
        if(tokens<=128)return std::min<int64_t>(std::max<std::size_t>(1,tokens),limit);
#ifdef TYLOINFER_NATIVE_CUDA
        return budget.chunk(tyloi::Device(tyloi::kCUDA, mfq_current_cuda_device()),
            tokens, limit, info.prefill_row_bytes, state.position() > 0,
            config.prefill_chunk_auto, generation_borrow(model, state, tokens, prepared ? &*prepared : nullptr));
#else
        return std::min<int64_t>(tokens,limit);
#endif
    }
    void prepare_prefill_workspace(std::size_t tokens) {
        const auto mode = weights.prefill_mode(tokens, state.position() > 0);
        const auto limit = config.generation.prefill_chunk_size;
        if(mode == ::tyloi::weights::Cache::Prefill::retained) {
            budget.prepare_prefill(tyloi::Device(tyloi::kCUDA, mfq_current_cuda_device()), tokens, limit, mode);
            return;
        }
#ifdef TYLOINFER_NATIVE_CUDA
        const bool streamed = mode == ::tyloi::weights::Cache::Prefill::streamed;
        const bool retire_decode = generation_prefill_begin(model, graph, streamed);
        budget.prepare_prefill(tyloi::Device(tyloi::kCUDA, mfq_current_cuda_device()),
            tokens, limit, mode, retire_decode);
#endif
    }
    bool prepare_snapshot() {
#ifdef TYLOINFER_NATIVE_CUDA
        if constexpr(requires { snapshot_size(state, mtp); }) {
            weights.prepare(::tyloi::weights::Cache::Stage::snapshot, tyloi::Device(tyloi::kCUDA, mode.primary), nullptr);
            const auto size=snapshot_size(state,mtp);
            return budget.snapshot(tyloi::Device(tyloi::kCUDA, mfq_current_cuda_device()),
                size.bytes, size.peak);
        }
#endif
        return true;
    }
    void invalidate_plan() { graph.invalidate(); }
    std::size_t prefix_boundary=0;
    void set_prefix_boundary(std::size_t tokens) { prefix_boundary=tokens; }
    bool has_hidden(const Hidden &value) const { return value && value->defined(); }
    void reset() {
        if (batching) return;
        state.reset(1);
        if constexpr (requires { state.predictor; }) {
            if (mtp) mtp->reset(*state.predictor,1);
        }
    }
    void release() {
        if (batching) return; // Each row owns its slot cleanup.
        trace_expert_cache("end");
        try { reset(); }
        catch (...) { graph.invalidate(); throw; }
        if (std::uncaught_exceptions()) graph.invalidate();
    }
    Generation plain(tyloi::engine::InferenceRequest& input, tyloi::engine::InferenceOutput& output,
                     size_t reused, size_t stable) {
        if (batching) return batching->generate(request_id, input, output);
        prepare_prefill_workspace(input.prompt.size()-reused);
        const auto compute_chunk=prefill_chunk(input.prompt.size()-reused);
        auto logical_chunk=compute_chunk;
        if constexpr(requires { prefill_span(model,state,input.prompt.size(),compute_chunk,reused,stable,false,false); }) {
            logical_chunk=prefill_span(model,state,input.prompt.size(),compute_chunk,reused,stable,
                prepared && prepared->transformed(),
                weights.prefill_mode(input.prompt.size()-reused, state.position() > 0) == ::tyloi::weights::Cache::Prefill::retained);
        }
        auto graph_options = config.decode_graph;
        graph_options.enabled &= graph_allowed;
        return tyloi::engine::generate_sequence(
            CudaGenerationOps<Model, State>(model, state, executor, collectives,
                mode, profiler, weights, serial_branches, report_memory, info, graph, input,
                prepared ? &*prepared : nullptr,
                graph_options,compute_chunk,&output,decode_ready ? &budget : nullptr,source),
            output, input.prompt.size(), reused, stable,logical_chunk);
    }
    Generation speculate(tyloi::engine::InferenceRequest &input, tyloi::engine::InferenceOutput &output,
        size_t reused, const std::shared_ptr<const ::tyloi::Tensor>& restored, Hidden *committed) {
        if constexpr (has_forward<Model, State> && !requires {
            replay(model, state, *executor, int64_t{});
        }) {
            throw std::invalid_argument("this CUDA model has no speculative predictor");
        } else {
            prepare_prefill_workspace(input.prompt.size()-reused);
            return ::run_mtp_generation(model, state, executor, collectives,
                mode, weights, info, source,
                *mtp, mtp_state,
                input,
                output,
                prefill_chunk(input.prompt.size()-reused),
                prepared ? &*prepared : nullptr,
                reused,
                restored,
                committed,prefix_boundary,decode_ready ? &budget : nullptr);
        }
    }
};

} // namespace generation_detail

template <typename Model, typename State, typename Predictor>
Generation generate(Model &model, State &state, tyloi::cuda::Executor* executor,
    tyloi::cuda::Collectives& collectives, tyloi::cuda::Executor::Mode mode,
    CudaProfiler& profiler, ::tyloi::weights::Cache& weights, tyloi::engine::Budget& budget,
    bool& serial_branches, bool report_memory, bool graph_allowed,
    std::function<void()> drain,
    const tyloi::models::Info &info, const tyloi::ModelSource* source,
    DecodeGraphCache &graph, tyloi::engine::SessionCache<CudaSessionOps> &cache,
    const tyloi::engine::Config &config, tyloi::engine::InferenceRequest &request,
    tyloi::engine::InferenceOutput &output, tyloi::engine::mtp::State &mtp_state,
    Predictor *mtp,
    std::optional<tyloi::models::Prepared> prepared,
    tyloi::engine::Batch* batching, std::string request_id,
    bool decode_ready) {
    if (mtp) {
        if constexpr (requires { state.predictor; }) {
            if (!state.predictor || !state.predictor->belongs_to(*mtp))
                throw std::invalid_argument("predictor state is unavailable or belongs to another model");
        } else {
            throw std::invalid_argument("model state has no predictor state");
        }
    }
    return tyloi::engine::generate_request(
        generation_detail::RequestOps<Model, State, Predictor>{model, state, executor, collectives, mode,
                                     profiler, weights, budget, serial_branches, report_memory, graph_allowed,
                                     std::move(drain), info, source, graph, cache, config, mtp_state,
                                     mtp,
                                     std::move(prepared), batching,
                                     std::move(request_id), decode_ready},
        request,
        output);
}
} // namespace tyloi::cuda::internal
