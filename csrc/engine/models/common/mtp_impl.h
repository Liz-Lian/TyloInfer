#pragma once
#include "engine/weights/workspace.h"

#include "env.h"
#include "engine/weights/routed_cache.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/decode_graph.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/generation/mtp/state.h"
#include "engine/models/common/causal_metadata.h"
#include "engine/models/common/prefetch.h"
#include "engine/models/common/mtp.h"
#include "engine/models/common/prompt.h"
#include "engine/backends/cuda/executor/sampling.h"
#include "engine/requests/policy.h"
#include "engine/requests/output.h"
#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"
#include "engine/models/common/mtp_ops.h"
#include "engine/models/common/generation_ops.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/models/input.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>
using mfq::cuda::internal::PrefillCudaTimer;

using namespace mfq::engine;

namespace mtp_detail {
using mfq::cuda::generation_workspace;
using mfq::cuda::mtp_stream_prompt_history;
using mfq::cuda::mtp_host_prompt_history;
using mfq::cuda::mtp_bootstrap_before_publication;
using mfq::cuda::mtp_retains_partial_target_prefix;
using mfq::cuda::mtp_history_merge_limit;
using mfq::cuda::mtp_history_flush_threshold;
using mfq::cuda::mtp_parked_history_flush_threshold;
using mfq::cuda::mtp_predictor_prefill_chunk_size;
using mfq::cuda::mtp_cache_only_prompt_history;
using mfq::cuda::mtp_initialize;
using mfq::cuda::mtp_target_forward;
using mfq::cuda::mtp_prefill;
using mfq::cuda::mtp_predictor_step;
using mfq::cuda::mtp_predictor_compact;
using mfq::cuda::mtp_predictor_history_step;
using mfq::cuda::mtp_begin_predictor_prefill;
using mfq::cuda::mtp_predictor_prefill_begin;
using mfq::cuda::mtp_predictor_prefill;

using Tensor = mfq_tensor_backend::Tensor;

template <typename Model, typename State>
constexpr bool has_forward =
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
  requires(Model& model, State& state, mfq::cuda::Input input,
           const mfq::models::ForwardOptions& options) {
    model.forward(std::move(input), state, options);
  } ||
#endif
  requires(
    Model& model, State& state, mfq::cuda::Executor& executor,
    mfq::cuda::Collectives& collectives, const mfq::cuda::Executor::Mode& mode,
    mfq::cuda::Input input) {
  forward(model, state, executor, collectives, mode, std::move(input));
};

template <typename Model, typename State>
constexpr bool has_replay = requires(
    Model& model, State& state, mfq::cuda::Executor& executor) {
  replay(model, state, executor, std::int64_t{});
};

template <typename Model, typename State>
Tensor hidden(Model& model, State& state, mfq::cuda::Executor* executor,
              mfq::cuda::Collectives& collectives, const mfq::cuda::Executor::Mode& mode,
              mfq::cuda::Input input) {
  static_assert(has_forward<Model, State>);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
  if constexpr (requires { model.forward(std::move(input), state, mfq::cuda::forward_options(mode)); }) {
    return model.forward(std::move(input), state, mfq::cuda::forward_options(mode));
  } else
#endif
  {
    TYLOINFER_RUNTIME_CHECK(executor != nullptr,
                            "CUDA model forward requires an executor");
    return forward(model, state, *executor, collectives, mode, std::move(input));
  }
}
template <typename Model, typename State>
void replay_target(Model& model, State& state, mfq::cuda::Executor* executor,
                   std::int64_t accepted) {
  static_assert(has_replay<Model, State>);
  TYLOINFER_RUNTIME_CHECK(executor != nullptr,
                          "CUDA model replay requires an executor");
  replay(model, state, *executor, accepted);
}

template <class Model, class State> struct CudaMtpOps {
  using Tensor = mfq_tensor_backend::Tensor;
  Model &model;
  State &state;
  mfq::cuda::Executor* executor;
  mfq::cuda::Collectives& collectives;
  const mfq::cuda::Executor::Mode mode;
  mfq::cuda::RoutedCache& weights;
  const mfq::models::CausalLmMetadata &metadata;
  MtpModule &mtp;
  mfq::engine::mtp::State &mtp_state;
  const mfq::cuda::PreparedPrompt *prepared;
  bool transformed_prompt;
  int position_axes;
  int64_t prefill_compute_chunk;
  InferenceOutput* output;
  std::function<void()> restore;
  // Destroy after local/result tensors, including cancellation of priming.
  std::unique_ptr<MoePrefillWorkspaceLoan> predictor_loan;
  const mfq_tensor_backend::TensorOptions options =
      mfq_tensor_backend::TensorOptions()
          .device(mfq_tensor_backend::kCUDA)
          .dtype(mfq_tensor_backend::kInt64);
  Tensor input_ids, counts;
  Tensor known_ids_host, known_ids_device;
  Tensor early_ple_ids_host;
  Tensor projected_target_hidden, projected_target_logits;
  Tensor projected_predictor_hidden, projected_predictor_logits;
  Tensor projected_target_compact,projected_predictor_compact;
  int graph_sampling_reuses=0;
  int32_t projected_predictor_greedy=-1;
  std::vector<int32_t> projected_target_greedy;
  std::optional<mfq::cuda::Sampler> sampler_;
  bool penalties = false;
  bool target_maintenance = false;
  std::uint64_t target_graph_captures = 0;
  std::uint64_t target_transfer_bytes=0;
  std::array<double,6> target_profile{};
  std::shared_ptr<MoeExpertCache> target_route_cache;
  static bool profile_target() {
    static const bool enabled=[] {
      const auto* option=mfq::cuda::env_lookup("TYLOINFER_PROFILE_MTP");
      return option && option[0]=='1';
    }();
    return enabled;
  }
  std::array<double,6> target_costs() const {
    std::array<double,6> result{};
    for(const auto& [name,value]:weights.metrics()) {
      if(name=="expert_route_wait_ms")result[0]=value;
      else if(name=="expert_host_fetch_ms")result[1]=value;
      else if(name=="expert_host_copy_ms")result[2]=value;
      else if(name=="ssd_expert_read_wait_ms")result[3]=value;
    }
    const auto* source = model.source.get();
    if(source) {
      const auto io=source->file_read_stats();
      result[4]=double(io.read_nanoseconds)/1e6;
      result[5]=double(io.physical_bytes)/1048576.0;
    }
    return result;
  }
  CudaMtpOps(CudaMtpOps&&)=default;
  ~CudaMtpOps() {
    if(target_route_cache)discard_moe_speculative_routes(target_route_cache);
  }
  void begin_target_routes(int rows) {
    begin_moe_speculative_routes(weights.moe,rows);
    target_route_cache=weights.moe;
  }
  void resolve_target_routes(int rows,bool cancelled) {
    if(!target_route_cache)return;
    if(cancelled)discard_moe_speculative_routes(target_route_cache);
    else commit_moe_speculative_routes(target_route_cache,rows);
    target_route_cache.reset();
  }
  bool take_target_maintenance() {
    const auto captures=moe_expert_graph_captures(weights.moe);
    const bool initialized=captures!=target_graph_captures;
    target_graph_captures=captures;
    return std::exchange(target_maintenance,false) || initialized;
  }
  bool prefetch_target_ids(int32_t token,const std::vector<int32_t>& drafts) {
    if constexpr(requires {model.adapter_prefetch_token_rows(model,early_ple_ids_host);}) {
      const auto* option=mfq::cuda::env_lookup("TYLOINFER_MTP_EARLY_PLE");
      if(option && option[0]=='0')return false;
      const int tokens=mtp.split_target_verification()?1:1+int(drafts.size());
      if(!early_ple_ids_host.defined() || early_ple_ids_host.size(1)<tokens)
        early_ple_ids_host=mfq_tensor_backend::empty({1,tokens},options.device(mfq_tensor_backend::kCPU));
      auto* ids=early_ple_ids_host.template data_ptr<int64_t>();ids[0]=token;
      for(int i=1;i<tokens;++i)ids[i]=drafts[std::size_t(i-1)];
      if(mfq::cuda::internal::prefetch_token_rows(model,early_ple_ids_host.narrow(1,0,tokens))) {
        static bool reported=false;
        if(!reported) {
          reported=true;std::cerr<<"runtime_mtp_early_token_rows enabled=1 prefix_tokens=1\n";
        }
        if(tokens>1 && tokens<64) {
          static uint64_t reported_prefixes=0;const auto bit=uint64_t(1)<<tokens;
          if(!(reported_prefixes&bit)) {
            reported_prefixes|=bit;
            std::cerr<<"runtime_mtp_token_row_draft_prefix enabled=1 prefix_tokens="<<tokens<<'\n';
          }
        }
        return true;
      }
    }
    return false;
  }
  void prefetch_target_first(int32_t token) {prefetch_target_ids(token,{});}
  void prefetch_target_prefix(int32_t token,const std::vector<int32_t>& drafts) {
    const auto* option=mfq::cuda::env_lookup("TYLOINFER_MTP_PLE_DRAFT_PREFIX");
    if(!option || option[0]!='0')prefetch_target_ids(token,drafts);
  }
  void prefetch_committed_token(int32_t token) {
    const auto* option=mfq::cuda::env_lookup("TYLOINFER_MTP_PLE_PUBLICATION");
    if(option && option[0]=='0')return;
    if(prefetch_target_ids(token,{})) {
      static bool reported=false;
      if(!reported) {
        reported=true;std::cerr<<"runtime_mtp_token_row_publication enabled=1\n";
      }
    }
  }
  bool stream_prompt_history(int64_t tokens) const {
    const auto* option=mfq::cuda::env_lookup("TYLOINFER_MTP_STREAM_PROMPT_HISTORY");
    return mtp_stream_prompt_history(model,tokens,option);
  }
  bool host_prompt_history(int64_t tokens) const {
    return mtp_host_prompt_history(model,tokens);
  }
  void trace_prompt_prefill(double target_ms,double predictor_ms,double workspace_ms) const {
    // Block predictors seed only their bounded target window and skip the
    // teacher-forced priming hook that restores workspace for chain predictors.
    if(!mtp.teacher_forced_prompt_prime() && restore)
      restore();
    std::cerr<<"runtime_mtp_prefill target_ms="<<target_ms<<" predictor_ms="<<predictor_ms
        <<" predictor_workspace_ms="<<workspace_ms<<" streamed="<<stream_prompt_history(input_ids.size(1))<<std::endl;
  }
  bool bootstrap_before_publication() const {
    const auto* option=mfq::cuda::env_lookup("TYLOINFER_MTP_EARLY_BOOTSTRAP");
    return mtp_bootstrap_before_publication(model,option);
  }
  bool retains_partial_target_prefix(int accepted) const {
    return mtp_retains_partial_target_prefix(model,mtp,accepted);
  }
  std::size_t predictor_history_merge_limit() const {
    return mtp_history_merge_limit(model);
  }
  std::size_t predictor_history_flush_threshold() const {
    const auto* option=mfq::cuda::env_lookup("TYLOINFER_MTP_HISTORY_EARLY_FLUSH");
    return mtp_history_flush_threshold(model,option);
  }
  std::size_t predictor_parked_history_flush_threshold() const {
    const auto* option=mfq::cuda::env_lookup("TYLOINFER_MTP_HISTORY_EARLY_FLUSH");
    return mtp_parked_history_flush_threshold(model,option);
  }
  void trace_cycle(uint64_t cycle,int depth,int accepted,double draft_ms,double verify_ms,double commit_ms) {
      const auto* option=mfq::cuda::env_lookup("TYLOINFER_PROFILE_MTP");
      const bool profile=option && option[0]=='1';
      if(profile || (cycle<=32 && draft_ms+verify_ms+commit_ms>=55.0)) {
        std::cerr<<(profile?"runtime_mtp_cycle_profile cycle=":"runtime_mtp_slow_cycle cycle=")<<cycle<<" depth="<<depth<<" accepted="<<accepted
            <<" draft_ms="<<draft_ms<<" verify_ms="<<verify_ms<<" commit_ms="<<commit_ms
            <<" target_transfer_mb="<<target_transfer_bytes/1048576.0;
        if(profile)
          std::cerr<<" target_route_ms="<<target_profile[0]<<" target_fetch_ms="<<target_profile[1]
              <<" target_host_copy_ms="<<target_profile[2]<<" target_ssd_wait_ms="<<target_profile[3]
              <<" target_parallel_read_ms="<<target_profile[4]<<" target_physical_read_mb="<<target_profile[5];
        std::cerr<<'\n';
      }
  }

  CudaMtpOps(Model &model, State &state, mfq::cuda::Executor* executor,
             mfq::cuda::Collectives& collectives, mfq::cuda::Executor::Mode mode,
             mfq::cuda::RoutedCache& weights,
             const mfq::models::CausalLmMetadata &metadata,
             MtpModule &mtp, mfq::engine::mtp::State &mtp_state,
             const mfq::cuda::PreparedPrompt *prepared,
             int64_t compute_chunk, InferenceOutput* output,
             std::function<void()> restore = {})
      : model(model), state(state), executor(executor), collectives(collectives), mode(mode),
        weights(weights), metadata(metadata), mtp(mtp), mtp_state(mtp_state), prepared(prepared),
        transformed_prompt(prepared && prepared->transformed()),
        position_axes(transformed_prompt && prepared->positions.defined() && prepared->positions.dim() == 2 &&
                              prepared->positions.size(0) == 3
                          ? 3
                          : 1), prefill_compute_chunk(compute_chunk), output(output),
        restore(std::move(restore)) {
    target_graph_captures=moe_expert_graph_captures(weights.moe);
  }
  void rollback_target(std::int64_t accepted) {
    if constexpr (has_replay<Model, State>)
      replay_target(model, state, executor, accepted);
    else
      model.rollback_speculative(accepted);
  }
  void initialize(const std::vector<int64_t> &prompt,
                  const MfqSamplingParams &sampling) {
    input_ids = ids_for(prompt);
    sampler_.emplace(sampling);
    penalties = sampler().has_penalties();
    mtp_initialize(model,*this);
    if (penalties) {
      counts = mfq::cuda::SamplingOps::token_counts(input_ids, metadata.vocab_size);
    }
  }
  auto &sampler() { return *sampler_; }
  bool captured_compact_requested() const {
    const auto* option=mfq::cuda::env_lookup("TYLOINFER_MTP_CONDITIONAL_TOPK");
    if(option && option[0]=='0')return true;
    const auto& parameters=sampler_->params();
    return !sampler_->greedy() && !penalties && parameters.top_k>0 && parameters.top_k<=64;
  }
  static void add_counts(Tensor counts, const Tensor &ids) {
    sample_token_counts_add_cuda(counts, ids);
  }
  static bool defined(const Tensor &value) { return value.defined(); }
  static int64_t size(const Tensor &value, int axis) {
    return value.size(axis);
  }
  static int rank(const Tensor &value) { return value.dim(); }
  static Tensor clone(const Tensor &value) {
    return value.is_cuda()?value.clone():value.to(mfq_tensor_backend::kCUDA);
  }
  static Tensor slice(const Tensor &value, int axis, int64_t offset,
                      int64_t length) {
    return value.narrow(axis, offset, length).contiguous();
  }
  static Tensor reshape(Tensor value, std::initializer_list<int64_t> shape) {
    return value.reshape(shape);
  }
  static Tensor concatenate(const std::vector<Tensor> &values, int axis) {
    if(std::any_of(values.begin(),values.end(),[](const auto& value){return !value.is_cuda();})) {
      auto host=values;
      for(auto& value:host)value=value.to(mfq_tensor_backend::kCPU);
      return mfq_tensor_backend::cat(host,axis).contiguous();
    }
    return mfq_tensor_backend::cat(values, axis).contiguous();
  }
  Tensor target_forward(Tensor ids, Tensor *raw = nullptr,
                        int64_t confirmed = 0) {
    projected_target_hidden = {}; projected_target_logits = {};
    projected_target_compact={};
    projected_target_greedy.clear();
    target_transfer_bytes = 0;
    const auto transfer_before=moe_expert_streamed_bytes(weights.moe);
    const auto costs_before=profile_target()?target_costs():std::array<double,6>{};
    const auto finish_profile=[&] {
      if(!profile_target())return;
      const auto costs_after=target_costs();
      for(std::size_t i=0;i<target_profile.size();++i)target_profile[i]=costs_after[i]-costs_before[i];
    };
    if(auto result=mtp_target_forward(model,*this,ids,raw,confirmed,transfer_before,finish_profile))
      return std::move(*result);
    Tensor hidden;
    if constexpr (has_forward<Model, State>) {
      mfq::cuda::Input input;
      input.ids = std::move(ids);
      input.raw_hidden = raw;
      input.confirmed = confirmed;
      hidden = mtp_detail::hidden(model, state, executor, collectives, mode, std::move(input));
    } else {
      hidden = model.hidden_forward(ids, mfq_nullopt, mfq_nullopt, nullptr,
                                    mfq_nullopt, raw, confirmed);
    }
    target_transfer_bytes=moe_expert_streamed_bytes(weights.moe)-transfer_before;
    finish_profile();
    return hidden;
  }
  Tensor target_suffix(Tensor ids, Tensor *raw) {
    if constexpr (requires(Model& value) { value.supports_suffix_speculation(); }) {
      TYLOINFER_RUNTIME_CHECK(model.supports_suffix_speculation(),
                              "model does not support speculative suffix forwarding");
    }
    if constexpr (requires(Model& value, Tensor value_ids, Tensor* value_raw) {
                    value.hidden_forward_speculative_suffix(std::move(value_ids), value_raw);
                  }) {
      projected_target_hidden = {}; projected_target_logits = {};
      projected_target_compact={};
      projected_target_greedy.clear();
      const auto transfer_before=moe_expert_streamed_bytes(weights.moe);
      const auto costs_before=profile_target()?target_costs():std::array<double,6>{};
      auto hidden=model.hidden_forward_speculative_suffix(ids, raw);
      target_transfer_bytes+=moe_expert_streamed_bytes(weights.moe)-transfer_before;
      if(profile_target()) {
        const auto costs_after=target_costs();
        for(std::size_t i=0;i<target_profile.size();++i)target_profile[i]+=costs_after[i]-costs_before[i];
      }
      return hidden;
    } else {
      throw std::runtime_error("model does not support speculative suffix forwarding");
    }
  }
  struct PrefillResult {
    Tensor hidden, raw;
    double elapsed;
  };
  double blockwise_prompt_context(Tensor hidden,int64_t position,bool seed) {
    PrefillCudaTimer timer;
    if(seed)mtp.seed_target_context(std::move(hidden),position);
    else {
      const auto start=position-hidden.size(1);
      mtp.append_target_context(std::move(hidden),start);
    }
    TYLOINFER_CUDA_CHECK(cudaEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
    return timer.elapsed_ms();
  }
  PrefillResult prefill(mfq::engine::PrefillChunk chunk) {
    PrefillCudaTimer timer;
    bool prefill_finished=false;
    const auto finish_prefill=[&] {
      if(prefill_finished)return;
      TYLOINFER_CUDA_CHECK(cudaEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
      prefill_finished=true;
    };
    Tensor raw;
    auto ids = input_ids.narrow(1, chunk.offset, chunk.count).contiguous();
    mfq::cuda::ImageMaskScope image_scope(model,prepared,chunk.offset,chunk.count);
    if(auto result=mtp_prefill(model,*this,chunk,ids,raw,timer,finish_prefill))
      return std::move(*result);
    MoePrefillWorkspaceLoan workspace(weights.moe);
    const bool host_history=host_prompt_history(chunk.count);
    const bool borrowed=transformed_prompt && generation_workspace(model,workspace,chunk.count);
    Tensor hidden;
    if (transformed_prompt) {
      auto embeddings = prepared->embeddings.narrow(1, chunk.offset, chunk.count)
                            .contiguous();
      auto positions = prepared->positions.defined()
                           ? MfqOptional<Tensor>(prepared->positions.narrow(
                                 -1, chunk.offset, chunk.count).contiguous())
                           : mfq_nullopt;
      if constexpr (has_forward<Model, State>) {
        mfq::cuda::Input input;
        input.ids = std::move(ids);
        input.embeddings = std::move(embeddings);
        if (positions) input.positions = std::move(*positions);
        input.advance = true;
        input.raw_hidden = &raw;
        hidden = mtp_detail::hidden(model, state, executor, collectives, mode, std::move(input));
      } else {
        hidden = model.hidden_forward_inputs(
            ids, std::move(embeddings), std::move(positions), mfq_nullopt, nullptr,
            mfq_nullopt, true, mfq_nullopt, &raw);
      }
    } else {
      hidden = target_forward(std::move(ids), &raw);
    }
    if(host_history)raw=raw.to(mfq_tensor_backend::kCPU);
    finish_prefill();
    if(borrowed) {
      workspace.suspend();
      hidden=hidden.narrow(1,hidden.size(1)-1,1).clone();
      if(!host_history)raw=raw.clone();
      workspace.finish();
    }
    return {std::move(hidden), std::move(raw), timer.elapsed_ms()};
  }
  Tensor ids_for(std::vector<int64_t> tokens) {
    known_ids_host = mfq_tensor_backend::tensor(tokens, options.device(mfq_tensor_backend::kCPU))
        .reshape({1, -1}).contiguous();
    known_ids_device = known_ids_host.to(mfq_tensor_backend::kCUDA);
    return known_ids_device;
  }
  Tensor decode_positions(int64_t cache_start, int64_t tokens) {
    TYLOINFER_RUNTIME_CHECK(tokens > 0, "CUDA MTP position span must be positive");
    const int64_t logical_start =
        cache_start +
        (transformed_prompt ? prepared->decode_position_delta : 0);
    TYLOINFER_RUNTIME_CHECK(logical_start >= 0 && logical_start + tokens <=
                                                metadata.max_position_embeddings,
                      "CUDA MTP position span exceeds context capacity");
    auto result = mfq_tensor_backend::arange(logical_start,
                                             logical_start + tokens, options);
    return position_axes == 3
               ? result.reshape({1, tokens}).expand({3, tokens}).contiguous()
               : result;
  }
  MtpStep predictor_step(Tensor hidden, Tensor ids, Tensor positions) {
    if(auto result=mtp_predictor_step(model,*this,hidden,ids,positions))
      return std::move(*result);
    auto result=positions.defined()
               ? mtp.step_positioned(std::move(hidden), std::move(ids),
                                     std::move(positions))
               : mtp.step(std::move(hidden), std::move(ids));
    const bool initialized=mtp.take_graph_initialization();
    target_maintenance=target_maintenance || initialized;
    projected_predictor_logits = result.sample_logits;
    projected_predictor_compact={};
    mtp_predictor_compact(model,*this,result);
    projected_predictor_greedy = result.greedy_token;
    projected_predictor_hidden = result.sample_logits.defined()
        ? result.sample_hidden.narrow(1, result.sample_hidden.size(1)-1, 1).contiguous() : Tensor{};
    return result;
  }
  void predictor_history_step(Tensor hidden, Tensor ids, Tensor positions,bool scalar_chain=false) {
    if(mtp_predictor_history_step(model,*this,hidden,ids,positions,scalar_chain))return;
    (void)predictor_step(std::move(hidden),std::move(ids),std::move(positions));
  }
  int64_t predictor_prefill_chunk_size(int64_t requested) const {
    return mtp_predictor_prefill_chunk_size(model,requested);
  }
  bool cache_only_prompt_history() const {
    return mtp_cache_only_prompt_history(model,*this);
  }
  double begin_predictor_prefill(int64_t pairs) {
    const auto started=std::chrono::steady_clock::now();
    return mtp_begin_predictor_prefill(model,*this,pairs,started);
  }
  double end_predictor_prefill() {
    if(predictor_loan){predictor_loan->finish();predictor_loan.reset();}
    if (restore) restore();
    // Every predictor tile has already synchronized its compute-end event.
    // Returning the expert loan is TTFT maintenance, not prompt computation.
    return 0;
  }
  double predictor_prefill(Tensor hidden, Tensor ids, Tensor positions) {
    mtp_predictor_prefill_begin(model,*this,ids);
    PrefillCudaTimer timer;
    bool prefill_finished=false;
    if(!cache_only_prompt_history())hidden=hidden.to(mfq_tensor_backend::kCUDA);
    mtp_predictor_prefill(model,*this,hidden,ids,positions,timer,prefill_finished);
    if(!prefill_finished)
      TYLOINFER_CUDA_CHECK(cudaEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
    return timer.elapsed_ms();
  }
  int32_t sample_normal(Tensor logits, Tensor token_counts) {
    if(sampler().greedy() && !penalties) {
      if(projected_predictor_greedy>=0 && projected_predictor_logits.defined() &&
          logits.data_ptr()==projected_predictor_logits.data_ptr() && logits.numel()==projected_predictor_logits.numel())
        return projected_predictor_greedy;
      if(projected_target_greedy.size()==1 && projected_target_logits.defined() &&
          logits.data_ptr()==projected_target_logits.data_ptr() && logits.numel()==projected_target_logits.numel())
        return projected_target_greedy.front();
    }
    if (penalties)
      logits = logits.clone();
    return static_cast<int32_t>(
        mfq::engine::sample_logits(sampler(), std::move(logits), token_counts)
            .template item<int64_t>());
  }
  std::vector<int32_t> greedy_tokens(Tensor logits) {
    TYLOINFER_RUNTIME_CHECK(!penalties && sampler().greedy() && logits.dim() == 2,
                      "batched greedy MTP sampling geometry is invalid");
    if(!projected_target_greedy.empty() && projected_target_logits.defined() &&
        logits.data_ptr()==projected_target_logits.data_ptr() && logits.numel()==projected_target_logits.numel())
      return projected_target_greedy;
    auto host = sampler().ops().sample_greedy(logits.contiguous()).cpu().contiguous();
    const auto* tokens = host.template data_ptr<int64_t>();
    return std::vector<int32_t>(tokens, tokens + host.numel());
  }
  int32_t sample_constrained(Tensor logits, Tensor token_counts,
                             const MfqTokenConstraintPtr &constraint) {
    if(!constraint && sampler().greedy() && !penalties) return sample_normal(logits,token_counts);
    if (penalties)
      logits = logits.clone();
    auto sampler_constraint =
        constraint ? constraint->clone() : MfqTokenConstraintPtr{};
    return static_cast<int32_t>(
        mfq::engine::sample_logits(sampler(), std::move(logits), token_counts,
                                 sampler_constraint)
            .template item<int64_t>());
  }
  std::vector<float> probabilities(Tensor logits, Tensor token_counts,
                                   const MfqSamplingParams &parameters) {
    logits = logits.contiguous().reshape({1, -1});
    if (penalties) {
      logits = logits.clone();
      sample_apply_penalties_cuda(
          logits, token_counts, parameters.presence_penalty,
          parameters.frequency_penalty, parameters.repetition_penalty);
    }
    return mfq::cuda::SamplingOps::probabilities(std::move(logits), parameters);
  }
  Tensor captured_compact(const Tensor& logits,const MfqSamplingParams& parameters) {
    for(const auto& pair:std::array<std::pair<Tensor,Tensor>,2>{{
        {projected_predictor_logits,projected_predictor_compact},{projected_target_logits,projected_target_compact}}})
      if(pair.first.defined() && pair.second.defined() && pair.first.data_ptr()==logits.data_ptr() &&
          pair.first.numel()==logits.numel() && pair.second.size(0)==logits.size(0)) {
        if(++graph_sampling_reuses==1)std::cerr<<"runtime_mtp_graph_sampling captured_topk=64 host_rows="<<logits.size(0)<<'\n';
#ifdef TYLOINFER_MTP_VERIFY_GRAPH_TOPK
        static thread_local std::array<bool,33> checked{};
        if(!checked[logits.size(0)]) {
            auto expected=mfq::cuda::SamplingOps::compact_radix_rows(logits,parameters);
            auto actual=mfq::cuda::SamplingOps::compact_rows_from_host(pair.second,parameters);
            TYLOINFER_RUNTIME_CHECK(expected.size()==actual.size(),"captured MTP top-k row count differs");
            for(std::size_t row=0;row<actual.size();++row) {
                TYLOINFER_RUNTIME_CHECK(expected[row].tokens==actual[row].tokens,"captured MTP top-k token IDs differ");
                for(std::size_t i=0;i<actual[row].probabilities.size();++i)
                    TYLOINFER_RUNTIME_CHECK(std::abs(actual[row].probabilities[i]-expected[row].probabilities[i])<=1e-6,
                        "captured MTP top-k probabilities differ");
            }
            checked[logits.size(0)]=true;target_maintenance=true;
            std::cerr<<"runtime_mtp_graph_sampling_check rows="<<logits.size(0)<<" PASS\n";
        }
#endif
        return pair.second;
      }
    return {};
  }
  mfq::engine::mtp::CompactDistribution
  compact_probabilities(Tensor logits, Tensor token_counts,
                        const MfqSamplingParams &parameters) {
    TYLOINFER_RUNTIME_CHECK(parameters.top_k > 0 && parameters.top_k <= 64 &&
                          parameters.temperature > 0.0,
                      "compact CUDA MTP sampling requires top-k 1-64");
    logits = logits.contiguous().reshape({1, -1});
    if(!penalties)if(auto host=captured_compact(logits,parameters);host.defined())
        return std::move(mfq::cuda::SamplingOps::compact_rows_from_host(host,parameters).front());
    if (penalties) {
      logits = logits.clone();
      sample_apply_penalties_cuda(
          logits, token_counts, parameters.presence_penalty,
          parameters.frequency_penalty, parameters.repetition_penalty);
    }
    return mfq::cuda::SamplingOps::compact_probabilities(std::move(logits), parameters);
  }
  std::vector<mfq::engine::mtp::CompactDistribution>
  compact_probability_rows(Tensor logits, const MfqSamplingParams &parameters) {
    TYLOINFER_RUNTIME_CHECK(!penalties && logits.dim() == 2 && parameters.top_k > 0 &&
                          parameters.top_k <= 64 &&
                          parameters.temperature > 0.0,
                      "batched compact CUDA MTP sampling geometry is invalid");
    logits = logits.contiguous();
    if(auto host=captured_compact(logits,parameters);host.defined())return mfq::cuda::SamplingOps::compact_rows_from_host(host,parameters);
    return mfq::cuda::SamplingOps::compact_probability_rows(std::move(logits), parameters);
  }
  Tensor logits_for(Tensor normalized) {
    const auto matches = [&](const Tensor& hidden) {
        return hidden.defined() && hidden.data_ptr()==normalized.data_ptr() && hidden.sizes()==normalized.sizes();
    };
    if(matches(projected_predictor_hidden)) return projected_predictor_logits;
    if(matches(projected_target_hidden)) return projected_target_logits;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    if constexpr (requires(Model& value, Tensor value_hidden) {
                    value.logits(std::move(value_hidden), true);
                  }) {
      auto input = (mtp.preserve_output_dtype()
                        ? normalized
                        : normalized.to(mfq_tensor_backend::kFloat16))
                       .contiguous();
      return model.logits(std::move(input), true);
    } else
#endif
    if constexpr (requires(Model& value, mfq::cuda::Executor& value_executor, Tensor value_hidden) {
                    logits(value, value_executor, std::move(value_hidden), true);
                  }) {
      TYLOINFER_RUNTIME_CHECK(executor != nullptr,
                              "CUDA model logits requires an executor");
      auto input = (mtp.preserve_output_dtype()
                        ? normalized
                        : normalized.to(mfq_tensor_backend::kFloat16))
                       .contiguous();
      return logits(model, *executor, std::move(input), true);
    } else {
      auto input = (mtp.preserve_output_dtype()
                        ? normalized
                        : normalized.to(mfq_tensor_backend::kFloat16))
                       .contiguous();
      return model.logits_from_hidden(std::move(input));
    }
  }
};

} // namespace mtp_detail
