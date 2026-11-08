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
template<class Ops> void mtp_initialize(qwen4_exp::Model&, Ops& ops) {
    if(auto* predictor=dynamic_cast<qwen4_exp::Predictor*>(&ops.mtp))
        predictor->set_compact_graph_sampling(*ops.state.predictor,ops.captured_compact_requested());
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

template<class Ops, class Tensor>
std::optional<tyloi::models::MtpStep> mtp_predictor_step(qwen4_exp::Model& model, Ops& ops, Tensor& hidden, Tensor& ids, Tensor& positions) {
    auto& mtp=ops.mtp;
#ifdef TYLOINFER_MTP_VERIFY_HISTORY_TAIL
    auto& target_maintenance=ops.target_maintenance;
#endif
      if(auto* predictor=dynamic_cast<tyloi::cuda::qwen4_exp::Predictor*>(&mtp);
          predictor && predictor->layers.size()==1 && ops.state.reuse_decode_graphs &&
          ops.state.graph_registry && ids.size(1)>1 && ids.size(1)<=128) {
        const auto prefix=ids.size(1)-1;
#ifdef TYLOINFER_MTP_VERIFY_HISTORY_TAIL
        static thread_local std::array<bool,129> checked{};
        Tensor expected_hidden,expected_chain;
        const auto check_start=ops.state.predictor->position();
        if(!checked[ids.size(1)]) {
          const auto start=ops.state.predictor->position();
          auto expected=mtp.step(*ops.state.predictor,hidden,ids,positions);
          #ifdef TYLOINFER_NATIVE_CUDA
          expected_hidden=ops.clone(ops.slice(expected.sample_hidden,1,prefix,1));
          expected_chain=ops.clone(ops.slice(expected.chain_hidden,1,prefix,1));
#else
          expected_hidden=expected.sample_hidden.dense().narrow(1,prefix,1).clone();
          expected_chain=expected.chain_hidden.dense().narrow(1,prefix,1).clone();
#endif
          mtp.trim_cache_to(*ops.state.predictor,start);
        }
#endif
        // Earlier teacher-forced rows contribute only predictor attention KV.
        // Their FFN/readout has no consumer; draft from the final row alone.
        const auto* chain_option=tyloi::cuda::env_lookup("TYLOINFER_MTP_HISTORY_CHAIN");
        if(prefix>1 && (!chain_option || chain_option[0]!='0')) {
          ops.predictor_history_step(ops.slice(hidden,1,0,prefix),ops.slice(ids,1,0,prefix),
              positions.defined()?ops.slice(positions,-1,0,prefix):Tensor{},true);
          static bool reported=false;
          if(!reported) {
            reported=true;std::cerr<<"runtime_mtp_history_chain enabled=1 rows="<<prefix
                <<" scalar_projections=1\n";
          }
        } else for(int64_t row=0;row<prefix;++row)
            ops.predictor_history_step(ops.slice(hidden,1,row,1),ops.slice(ids,1,row,1),
                positions.defined()?ops.slice(positions,-1,row,1):Tensor{});
        auto result=ops.predictor_step(ops.slice(hidden,1,prefix,1),ops.slice(ids,1,prefix,1),
            positions.defined()?ops.slice(positions,-1,prefix,1):Tensor{});
#ifdef TYLOINFER_MTP_VERIFY_HISTORY_TAIL
        if(expected_hidden.defined()) {
          const auto difference=[&](const tyloi::models::MtpStep& reference) {
            double maximum=0;bool close=true;
            for(const auto& pair:std::array<std::pair<std::shared_ptr<tyloi::Tensor>,std::shared_ptr<tyloi::Tensor>>,2>{{
                    {result.sample_hidden,reference.sample_hidden},{result.chain_hidden,reference.chain_hidden}}}) {
              const auto actual=dynamic_cast<const DenseTensor&>(*pair.first).to(::tyloi::kCPU,::tyloi::kFloat32).contiguous();
              const auto expected=dynamic_cast<const DenseTensor&>(*pair.second).to(::tyloi::kCPU,::tyloi::kFloat32).contiguous();
              TYLOINFER_RUNTIME_CHECK(actual.sizes()==expected.sizes(),"MTP history-tail shape differs from full predictor");
              for(int64_t i=0;i<actual.numel();++i) {
                const auto error=std::abs(double(actual.template data_ptr<float>()[i])-expected.template data_ptr<float>()[i]);
                maximum=std::max(maximum,error);
                close=close && std::isfinite(error) && error<=.003+.003*std::abs(expected.template data_ptr<float>()[i]);
              }
            }
            return std::make_pair(maximum,close);
          };
          const auto [batch_error,batch_close]=difference({expected_hidden,expected_chain});
          double scalar_error=0;
          if(!batch_close) {
            mtp.trim_cache_to(*ops.state.predictor,check_start);
            tyloi::models::MtpStep scalar;
            for(int64_t row=0;row<ids.size(1);++row) {
              auto row_positions=positions.defined()?ops.slice(positions,-1,row,1):Tensor{};
              scalar=mtp.step(*ops.state.predictor,ops.slice(hidden,1,row,1),ops.slice(ids,1,row,1),
                              row_positions);
            }
            const auto [error,close]=difference(scalar);scalar_error=error;
            std::fprintf(stderr,"runtime_mtp_history_tail_scalar_check rows=%lld batch_max_error=%.8f scalar_max_error=%.8f close=%d\n",
                static_cast<long long>(ids.size(1)),batch_error,scalar_error,int(close));
            TYLOINFER_RUNTIME_CHECK(close,"MTP history-tail differs from scalar predictor");
          }
          checked[ids.size(1)]=true;target_maintenance=true;
          std::fprintf(stderr,"runtime_mtp_history_tail_check rows=%lld batch_max_error=%.8f scalar_max_error=%.8f PASS\n",
              static_cast<long long>(ids.size(1)),batch_error,scalar_error);
        }
#endif
        return result;
      }
      if(ops.state.reuse_decode_graphs && ops.state.graph_registry && ids.size(1)>4 && ids.size(1)<=128) {
        tyloi::models::MtpStep last;
        for(int64_t offset=0;offset<ids.size(1);offset+=4) {
          const auto rows=std::min<int64_t>(4,ids.size(1)-offset);
          auto part_positions=positions.defined()?ops.slice(positions,-1,offset,rows):Tensor{};
          if(offset+rows<ids.size(1))
            ops.predictor_history_step(ops.slice(hidden,1,offset,rows),ops.slice(ids,1,offset,rows),part_positions);
          else last=ops.predictor_step(ops.slice(hidden,1,offset,rows),ops.slice(ids,1,offset,rows),part_positions);
        }
        return last;
      }
    return std::nullopt;
}

template<class Ops>
void mtp_predictor_compact(qwen4_exp::Model&, Ops& ops, tyloi::models::MtpStep& result) {
    if(auto* predictor=dynamic_cast<qwen4_exp::Predictor*>(&ops.mtp);predictor && result.sample_logits && result.sample_logits.defined())
        ops.projected_predictor_compact=predictor->graph_compact_topk(*ops.state.predictor);
}

template<class Ops, class Tensor>
bool mtp_predictor_history_step(qwen4_exp::Model& model, Ops& ops, Tensor& hidden, Tensor& ids, Tensor& positions, bool scalar_chain) {
    if(auto* predictor=dynamic_cast<qwen4_exp::Predictor*>(&ops.mtp);
        predictor && ops.state.reuse_decode_graphs && ops.state.graph_registry && ids.size(1)<=128) {
        for(int64_t offset=0;offset<ids.size(1);offset+=4) {
            const auto rows=std::min<int64_t>(4,ids.size(1)-offset);
#ifdef TYLOINFER_NATIVE_CUDA
            auto h=ops.slice(hidden,1,offset,rows), i=ops.slice(ids,1,offset,rows);
            auto p=positions.defined()?ops.slice(positions,-1,offset,rows):Tensor{};
            predictor->history_step(*ops.state.predictor,h.dense(),i.dense(),p ? p.dense() : ::tyloi::DenseTensor{},scalar_chain);
#else
            predictor->history_step(*ops.state.predictor,ops.slice(hidden,1,offset,rows),ops.slice(ids,1,offset,rows),
                positions.defined()?ops.slice(positions,-1,offset,rows):Tensor{},scalar_chain);
#endif
        }
        ops.target_maintenance=ops.state.predictor->take_graph_initialization() || ops.target_maintenance;
        return true;
    }
    return false;
}

template<class Ops, class Started>
double mtp_begin_predictor_prefill(qwen4_exp::Model& model, Ops& ops, int64_t pairs, Started started) {
    if(ops.cache_only_prompt_history())return 0;
    if(pairs<=128 || !ops.state.reuse_decode_graphs || !ops.state.graph_registry)return 0;
    const auto rows=std::min<int64_t>(1024,pairs);
    const auto bytes=(std::size_t(1)<<30)+std::size_t(rows)*model.config.hidden*
        model.config.streams*sizeof(float)*8;
    auto loan=std::make_unique<tyloi::weights::Loan>(ops.weights);
    if(loan->predictor(bytes))ops.predictor_loan=std::move(loan);
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
}

template<class Ops, class Tensor>
void mtp_predictor_prefill_begin(qwen4_exp::Model& model, Ops& ops, Tensor& ids) {
    if(!ops.cache_only_prompt_history() && ops.state.reuse_decode_graphs && ops.state.graph_registry && ids.size(1)<=128)
        ops.end_predictor_prefill();
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
    } else if(ops.state.reuse_decode_graphs && ops.state.graph_registry && ids.size(1)<=128) {
        for(int64_t offset=0;offset<ids.size(1);offset+=4) {
            const auto rows=std::min<int64_t>(4,ids.size(1)-offset);
            ops.predictor_history_step(ops.slice(hidden,1,offset,rows),ops.slice(ids,1,offset,rows),
                positions.defined()?ops.slice(positions,-1,offset,rows):Tensor{});
        }
    } else (void)ops.predictor_step(std::move(hidden),std::move(ids),std::move(positions));
}

} // namespace tyloi::cuda
