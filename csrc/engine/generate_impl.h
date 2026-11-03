#pragma once

#include "engine/backends/cuda/native/env.h"
#include "engine/generate.h"
#include "engine/models/common/generation_impl.h"
#include "engine/generation/generation.h"
#include "engine/generation/prefill_budget.h"
#include "engine/models/common/session/cache.h"
#include "engine/models/source.h"
#include "engine/weights/routed_cache.h"
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

namespace mfq::cuda::internal {
using mfq::engine::Generation;
namespace generation_detail {

template <class Model, class State, class Predictor> struct RequestOps {
    using Hidden = mfq_tensor_backend::Tensor;
    using Restore = TextSessionRestore;
    Model &model;
    State &state;
    mfq::cuda::Executor* executor;
    mfq::cuda::Collectives& collectives;
    const Executor::Mode mode;
    CudaProfiler& profiler;
    RoutedCache& weights;
    bool& serial_branches;
    bool report_memory;
    bool graph_allowed;
    std::function<void()> drain;
    const mfq::models::Info &info;
    const mfq::ModelSource* source;
    DecodeGraphCache &graph;
    TextSessionCache &cache;
    const CudaRuntimeConfig &config;
    mfq::engine::mtp::State &mtp_state;
    Predictor *mtp;
    std::optional<PreparedPrompt> prepared;
    mfq::engine::Batch* batching;
    std::string request_id;
    std::function<void()> restore;

    bool batched() const { return batching != nullptr; }
    void trace_expert_cache(const char* boundary, bool wait = true) {
        // Request/first-token boundaries only: existing counters need no
        // events or per-token synchronization. Full summaries drain exchanges.
        static const bool enabled=[] {
            const char* value=mfq::cuda::env_lookup("TYLOINFER_TRACE_MOE_SUMMARY");
            return value && value[0] && value[0]!='0';
        }();
        if (!enabled || batching || std::uncaught_exceptions()) return;
        std::cerr << "moe_request_cache_stats boundary=" << boundary << '\n';
        if(wait)print_moe_expert_cache_stats(weights.moe,std::cerr);
        // The first-token snapshot must not wait for exchanges or introduce
        // CUDA synchronization. Subtract it from the end snapshot to isolate
        // steady decode I/O from the much larger prefill scan.
        std::cerr << "moe_phase_metrics boundary=" << boundary;
        for(const auto& [name,value]:weights.metrics()) {
            if(name=="ssd_expert_read_wait_ms" || name=="expert_route_wait_ms" ||
               name=="expert_host_fetch_ms" || name=="expert_host_copy_ms" || name=="expert_transfer_cache_prepare_ms" || name=="ssd_expert_bytes_read" ||
               name=="ssd_expert_read_count" || name=="ram_expert_cache_hits" ||
               name=="ram_expert_cache_misses" || name=="ram_expert_pcie_read_bytes" ||
               name=="expert_logical_transfer_bytes")std::cerr << ' ' << name << '=' << value;
        }
        std::cerr << '\n';
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
        generation_request(model,*this,input);
#endif
    }
    void reserve_workspace(std::size_t bytes) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        {
            reserve_moe_workspace(weights.moe,bytes);
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
        return state.position()>0 && tokens<=128 &&
            moe_expert_cache_ssd_experts(weights.moe);
    }
    int64_t prefill_chunk(std::size_t tokens) {
        // Reuse ordinary/MTP widths before allocating wider temporary graphs
        // whose activation storage would displace persistent expert images.
        const auto limit=config.generation.prefill_chunk_size;
        if(short_streamed_prefill(tokens))return std::min<int64_t>(4,limit);
        if(tokens<=128)return std::min<int64_t>(std::max<std::size_t>(1,tokens),limit);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        const bool allow_borrow=generation_borrow(model,tokens,prepared?&*prepared:nullptr);
        auto budget=moe_prefill_workspace_budget(weights.moe,allow_borrow);
        const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
        auto available=mfq::engine::prefill_budget_add(
            mfq::engine::prefill_budget_add(context->available_memory(),context->workspace_available()),
            budget.lendable_bytes);
        const auto row_bytes=info.prefill_row_bytes;
        const auto sequence_bytes=allow_borrow
            ? std::size_t(tokens<=16384?tokens:std::min<std::size_t>(tokens,limit))*row_bytes*3/8 : 0;
        const auto reserve=moe_expert_cache_ssd_experts(weights.moe)
            ? std::max(std::size_t(1)<<30,mfq::cuda::runtime_options::workspace_reserve_bytes())
            : std::size_t(1)<<30;
        const auto fixed=mfq::engine::prefill_budget_add(reserve+sequence_bytes,budget.transfer_bytes);
        const auto wanted=std::min<int64_t>({int64_t(tokens),limit,mfq::engine::max_prefill_compute_chunk});
        const auto required=mfq::engine::prefill_budget_add(fixed,std::size_t(wanted)*row_bytes);
        if(required>available) {
            // Expert overflow in the unused KV budget is revocable too. Ask
            // its existing owner to return pages before reducing the tile.
            reserve_workspace(required>budget.lendable_bytes?required-budget.lendable_bytes:0);
            budget=moe_prefill_workspace_budget(weights.moe,allow_borrow);
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
        const bool streamed=tokens>8 && moe_expert_cache_ssd_experts(weights.moe);
        generation_prefill_begin(model,graph,streamed);
        const bool retained=streamed && lend_moe_decode_cache_for_prefill(weights.moe,
            mfq::cuda::runtime_options::workspace_reserve_bytes());
        generation_prefill_end(model,*this,streamed,retained);
#endif
        if(tokens>1024 || moe_expert_cache_ssd_experts(weights.moe))
            reserve_workspace(mfq::cuda::runtime_options::workspace_reserve_bytes());
    }
    bool prepare_snapshot() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if constexpr(requires { snapshot_size(state, mtp); }) {
            // Retire prefill buffers before copying KV/recurrent state. Evict
            // old hot entries before allocating the incoming checkpoint.
            release_moe_prefill_buffers(weights.moe);
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
        if constexpr(requires { prefill_span(model,input.prompt.size(),compute_chunk,reused,stable,false,false); }) {
            logical_chunk=prefill_span(model,input.prompt.size(),compute_chunk,reused,stable,
                prepared && prepared->transformed(),short_streamed_prefill(input.prompt.size()-reused));
        }
        auto graph_options = config.decode_graph;
        graph_options.enabled &= graph_allowed;
        return mfq::engine::generate_sequence(
            CudaGenerationOps<Model, State>(model, state, executor, collectives,
                mode, profiler, weights, serial_branches, report_memory, info, graph, input,
                prepared ? &*prepared : nullptr,
                graph_options,compute_chunk,&output,restore),
            output, input.prompt.size(), reused, stable,logical_chunk);
    }
    Generation speculate(mfq::engine::InferenceRequest &input, mfq::engine::InferenceOutput &output,
        size_t reused, Hidden restored, Hidden *committed) {
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
                std::move(restored),
                committed,prefix_boundary,restore);
        }
    }
};

} // namespace generation_detail

template <typename Model, typename State, typename Predictor>
Generation generate(Model &model, State &state, mfq::cuda::Executor* executor,
    mfq::cuda::Collectives& collectives, mfq::cuda::Executor::Mode mode,
    CudaProfiler& profiler, mfq::cuda::RoutedCache& weights,
    bool& serial_branches, bool report_memory, bool graph_allowed,
    std::function<void()> drain,
    const mfq::models::Info &info, const mfq::ModelSource* source,
    DecodeGraphCache &graph, TextSessionCache &cache,
    const CudaRuntimeConfig &config, mfq::engine::InferenceRequest &request,
    mfq::engine::InferenceOutput &output, mfq::engine::mtp::State &mtp_state,
    Predictor *mtp,
    std::optional<PreparedPrompt> prepared,
    mfq::engine::Batch* batching, std::string request_id,
    std::function<void()> restore) {
    return mfq::engine::generate_request(
        generation_detail::RequestOps<Model, State, Predictor>{model, state, executor, collectives, mode,
                                     profiler, weights, serial_branches, report_memory, graph_allowed,
                                     std::move(drain), info, source, graph, cache, config, mtp_state,
                                     mtp,
                                     std::move(prepared), batching,
                                     std::move(request_id), std::move(restore)},
        request,
        output);
}
} // namespace mfq::cuda::internal
