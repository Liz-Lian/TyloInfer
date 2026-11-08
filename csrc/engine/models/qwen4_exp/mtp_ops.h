#pragma once

#include "engine/weights/cache.h"
#include "engine/generation/mtp/ops.h"
#include "engine/models/qwen4_exp/generation.h"
#include "engine/models/qwen4_exp/predictor.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/native/profile.h"
#include "env.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace tyloi::cuda {

inline bool mtp_stream_prompt_history(qwen4_exp::Model&, int64_t tokens, const char* option) {
    return tokens>16384 && (!option || option[0]!='0');
}
inline bool mtp_host_prompt_history(qwen4_exp::Model& model, int64_t tokens) {
    const auto* option=env_lookup("TYLOINFER_MTP_HOST_PROMPT_HISTORY");
    if(option)return option[0]!='0';
    const auto bytes=std::size_t(tokens)*model.metadata.hidden_size*model.metadata.hc_mult*sizeof(uint16_t);
    return bytes+(std::size_t(32)<<20)>default_context(mfq_current_cuda_device())->available_memory();
}
inline bool mtp_bootstrap_before_publication(qwen4_exp::Model&, const char* option) { return !option || option[0]!='0'; }
template<class Mtp> bool mtp_retains_partial_target_prefix(qwen4_exp::Model& model, qwen4_exp::State& state, Mtp& mtp, int accepted) {
    if(state.speculative_prefix_retention && accepted>=0)return true;
    return mtp.retains_partial_target_prefix();
}
inline std::size_t mtp_history_merge_limit(qwen4_exp::Model&) { return 128; }
inline std::size_t mtp_history_flush_threshold(qwen4_exp::Model&, const char* option) { return !option || option[0]!='0'?4:512; }
inline std::size_t mtp_parked_history_flush_threshold(qwen4_exp::Model&, const char* option) { return !option || option[0]!='0'?16:512; }
inline int64_t mtp_predictor_prefill_chunk_size(qwen4_exp::Model&, int64_t requested) { return std::min<int64_t>(1024,requested); }
template<class Ops> bool mtp_cache_only_prompt_history(qwen4_exp::Model&, Ops& ops) {
    const auto* option=env_lookup("TYLOINFER_MTP_CACHE_ONLY_PRIME");
    const auto* predictor=dynamic_cast<const qwen4_exp::Predictor*>(&ops.mtp);
    return (!option || option[0]!='0') && predictor && predictor->layers.size()==1;
}
template<class Ops, class Tensor, class Finish>
std::optional<Tensor> mtp_target_forward(qwen4_exp::Model& model, Ops& ops, Tensor& ids, Tensor* raw,
                                         int64_t confirmed, std::uint64_t transfer_before, const Finish& finish_profile) {
    if(!ops.transformed_prompt) {
        bool initialized=false;
#ifdef TYLOINFER_NATIVE_CUDA
        const auto host_ids = ops.same_view(ids,ops.known_ids_device) ? ops.known_ids_host : Tensor{};
#else
        const auto host_ids = ops.known_ids_device.defined() &&
            ids.data_ptr() == ops.known_ids_device.data_ptr() && ids.sizes() == ops.known_ids_device.sizes()
            ? ops.known_ids_host : Tensor{};
#endif
        ::tyloi::Tensor raw_result, logits;
        if(auto hidden=qwen4_decode_graph_hidden(model,ops.state,ids,raw ? &raw_result : nullptr,confirmed,&initialized,host_ids,&logits,&ops.projected_target_greedy,ops.captured_compact_requested())) {
            ops.target_transfer_bytes=ops.weights.activity().streamed_bytes-transfer_before;
            ops.target_maintenance=ops.target_maintenance || initialized;
#ifdef TYLOINFER_NATIVE_CUDA
            if (raw) *raw = raw_result;
            ops.projected_target_hidden = *hidden; ops.projected_target_logits = logits;
            ops.projected_target_compact=qwen4_decode_graph_compact_topk(model,ops.state);
            finish_profile();
            return *hidden;
#else
            if (raw) *raw = raw_result.dense();
            ops.projected_target_hidden = hidden->dense(); ops.projected_target_logits = logits.dense();
            ops.projected_target_compact=qwen4_decode_graph_compact_topk(model,ops.state).dense();
            finish_profile();
            return hidden->dense();
#endif
        }
    }
    return std::nullopt;
}

template<class Ops, class Chunk, class Tensor, class Timer, class Finish>
std::optional<typename Ops::PrefillResult> mtp_prefill(qwen4_exp::Model& model, Ops& ops, const Chunk& chunk,
                                                      Tensor& ids, Tensor& raw, Timer& timer, const Finish& finish_prefill) {
    if(!ops.transformed_prompt && ops.state.reuse_decode_graphs && ops.state.graph_registry && chunk.count<=128) {
        std::vector<Tensor> hidden_rows,raw_rows;
        for(int64_t offset=0;offset<chunk.count;) {
            const auto rows=ops.state.cache_pos==0?int64_t(1):std::min<int64_t>(4,chunk.count-offset);
            Tensor part;
            hidden_rows.push_back(ops.clone(ops.target_forward(ops.slice(ids,1,offset,rows),&part,rows>1?1:0)));
            raw_rows.push_back(std::move(part));
            if(rows>1)ops.state.commit_speculative();
            offset+=rows;
        }
        auto hidden=hidden_rows.size()==1?std::move(hidden_rows.front()):ops.concatenate(hidden_rows,1);
        raw=raw_rows.size()==1?std::move(raw_rows.front()):ops.concatenate(raw_rows,1);
        TYLOINFER_CUDA_CHECK(cudaEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
        return typename Ops::PrefillResult{std::move(hidden),std::move(raw),timer.elapsed_ms()};
    }
    if(!ops.transformed_prompt && qwen4_layer_major_prefill_eligible(model,ops.state,chunk.count)) {
        const bool host_history=ops.host_prompt_history(chunk.count);
        ::tyloi::Tensor raw_result;
        auto hidden=qwen4_layer_major_prefill_hidden(model,ops.state,ids,ops.prefill_compute_chunk,true,
            [&] { return ops.output && ops.output->stopped(); }, &raw_result,host_history,finish_prefill);
#ifdef TYLOINFER_NATIVE_CUDA
        raw = std::move(raw_result);
        finish_prefill();
        return typename Ops::PrefillResult{std::move(hidden),std::move(raw),timer.elapsed_ms()};
#else
        if (raw_result) raw = raw_result.dense();
        finish_prefill();
        return typename Ops::PrefillResult{hidden ? hidden.dense() : Tensor{},std::move(raw),timer.elapsed_ms()};
#endif
    }
    return std::nullopt;
}

template<class Ops, class Tensor, class Timer>
void mtp_predictor_prefill(qwen4_exp::Model& model, Ops& ops, Tensor& hidden, Tensor& ids, Tensor& positions,
                           Timer& timer, bool& prefill_finished) {
    if(ops.cache_only_prompt_history()) {
        auto& predictor=static_cast<qwen4_exp::Predictor&>(ops.mtp);
        tyloi::weights::Loan workspace(ops.weights);
        if(ids.size(1)>128)workspace.begin((std::size_t(64)<<20)+
            std::size_t(ids.size(1))*model.metadata.hidden_size*model.metadata.hc_mult*sizeof(float)*12);
        {
#ifdef TYLOINFER_NATIVE_CUDA
            auto resident_hidden=tyloi::ops::copy(hidden,tyloi::Device(tyloi::kCUDA,ops.mode.primary));
            (void)predictor.evaluate(*ops.state.predictor,resident_hidden.dense(),ids.dense(),0,true,
                                     positions ? positions.dense() : ::tyloi::DenseTensor{}, {},true);
#else
            auto resident_hidden=hidden.to(::tyloi::kCUDA);
            (void)predictor.evaluate(*ops.state.predictor,resident_hidden,ids,0,true,positions,{},true);
#endif
        }
        TYLOINFER_CUDA_CHECK(cudaEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
        prefill_finished=true;
        workspace.finish();
    } else (void)ops.predictor_step(std::move(hidden),std::move(ids),std::move(positions));
}

} // namespace tyloi::cuda
