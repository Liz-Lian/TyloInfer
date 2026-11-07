#pragma once

#include "engine/backends/cuda/native/env.h"
#include "engine/generate.h"
#include "engine/budget.h"
#include "engine/models/common/generation_impl.h"
#include "engine/generation/generation.h"
#include "engine/models/common/session/cache.h"
#include "engine/models/source.h"
#include "engine/weights/cache_internal.h"
#include "engine/weights/workspace.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/native/profile.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

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
    const tyloi::engine::RuntimeConfig &config;
    tyloi::engine::mtp::State &mtp_state;
    Predictor *mtp;
    std::optional<PreparedPrompt> prepared;
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
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        generation_request(model,*this,input);
#endif
    }
    bool short_streamed_prefill(std::size_t tokens) const {
        // A short continuation can reuse the bounded MFE path without
        // returning its expert images and captured graphs for a large PP arena.
        return state.position()>0 && tokens<=128 &&
            moe_expert_cache_ssd_experts(::tyloi::weights::detail::cache(weights).moe);
    }
    int64_t prefill_chunk(std::size_t tokens) {
        // Reuse ordinary/MTP widths before allocating wider temporary graphs
        // whose activation storage would displace persistent expert images.
        const auto limit=config.generation.prefill_chunk_size;
        if(short_streamed_prefill(tokens))return std::min<int64_t>(4,limit);
        if(tokens<=128)return std::min<int64_t>(std::max<std::size_t>(1,tokens),limit);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        return budget.chunk(tyloi::Device(tyloi::kCUDA, mfq_current_cuda_device()),
            tokens, limit, info.prefill_row_bytes, state.position() > 0,
            config.prefill_chunk_auto, generation_borrow(model, tokens, prepared ? &*prepared : nullptr));
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
        const bool streamed=tokens>8 && moe_expert_cache_ssd_experts(::tyloi::weights::detail::cache(weights).moe);
        generation_prefill_begin(model,graph,streamed);
        const bool retained=streamed && lend_moe_decode_cache_for_prefill(::tyloi::weights::detail::cache(weights).moe,
            tyloi::cuda::runtime_options::workspace_reserve_bytes());
        generation_prefill_end(model,*this,streamed,retained);
        if(tokens>1024 || moe_expert_cache_ssd_experts(::tyloi::weights::detail::cache(weights).moe))
            budget.workspace(tyloi::Device(tyloi::kCUDA, mfq_current_cuda_device()),
                tyloi::cuda::runtime_options::workspace_reserve_bytes());
#endif
    }
    bool prepare_snapshot() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if constexpr(requires { snapshot_size(state, mtp); }) {
            // Retire prefill buffers before copying KV/recurrent state. Evict
            // old hot entries before allocating the incoming checkpoint.
            release_moe_prefill_buffers(::tyloi::weights::detail::cache(weights).moe);
            // A prefix stores occupied rows, not the full preallocated context
            // or graph-owned speculative checkpoints.
            const auto size=snapshot_size(state,mtp);
            const auto incoming=size.bytes;
            std::size_t hot=0,limit=0;
            for(const auto& metric:cache.metrics()) {
                if(metric.first=="prefix_cache_hot_bytes")hot=std::size_t(metric.second);
                if(metric.first=="prefix_cache_max_bytes")limit=std::size_t(metric.second);
            }
            const auto target=limit>incoming?limit-incoming:0;
            if(hot>target)cache.trim_hot(target);
            const auto needed=size.peak+(std::size_t(64)<<20);
            budget.workspace(tyloi::Device(tyloi::kCUDA, mfq_current_cuda_device()), needed);
            const auto available=tyloi::cuda::default_context(mfq_current_cuda_device())->available_memory();
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
    bool has_hidden(const Hidden &value) const { return value && value->defined(); }
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
    Generation plain(tyloi::engine::InferenceRequest& input, tyloi::engine::InferenceOutput& output,
                     size_t reused, size_t stable) {
        if (batching) return batching->generate(request_id, input, output);
        prepare_prefill_workspace(input.prompt.size()-reused);
        const auto compute_chunk=prefill_chunk(input.prompt.size()-reused);
        auto logical_chunk=compute_chunk;
        if constexpr(requires { prefill_span(model,input.prompt.size(),compute_chunk,reused,stable,false,false); }) {
            logical_chunk=prefill_span(model,input.prompt.size(),compute_chunk,reused,stable,
                prepared && prepared->transformed(),short_streamed_prefill(input.prompt.size()-reused));
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
    const tyloi::engine::RuntimeConfig &config, tyloi::engine::InferenceRequest &request,
    tyloi::engine::InferenceOutput &output, tyloi::engine::mtp::State &mtp_state,
    Predictor *mtp,
    std::optional<PreparedPrompt> prepared,
    tyloi::engine::Batch* batching, std::string request_id,
    bool decode_ready) {
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
